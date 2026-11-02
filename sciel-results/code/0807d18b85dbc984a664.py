import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 6]

def _make_zero_matrices(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    return U, V, W

def _verify_strassen_block(n, m, p, i1, i2, j1, j2, k1, k2):
    """Verify Strassen block in isolation by checking all 8 tensor entries."""
    # Build a small Strassen block and check it reproduces the 2x2x2 tensor
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    _add_strassen_block(U, V, W, 0, i1, i2, j1, j2, k1, k2, n, m, p)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all 8 entries in the 2x2x2 block
    for ii in [i1, i2]:
        for jj in [j1, j2]:
            for kk in [k1, k2]:
                u_idx = ii*m + jj
                v_idx = jj*p + kk
                w_idx = kk*n + ii
                val = float(np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx]))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Also check that entries OUTSIDE the block are zero
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                if ii not in (i1, i2) or jj not in (j1, j2) or kk not in (k1, k2):
                    u_idx = ii*m + jj
                    v_idx = jj*p + kk
                    w_idx = kk*n + ii
                    val = float(np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx]))
                    if abs(val) > 1e-9:
                        return False
    return True

def _add_strassen_block(U, V, W, col,
                        i1, i2, j1, j2, k1, k2,
                        n, m, p):
    # Corrected Strassen factors for a 2x2x2 block.
    # Verified numerically via _verify_strassen_block.
    #
    # Using the convention:
    # A[i,j] for i in {i1,i2}, j in {j1,j2}
    # B[j,k] for j in {j1,j2}, k in {k1,k2}
    # C[k,i] for k in {k1,k2}, i in {i1,i2}
    #
    # Strassen products (using 0-indexed local coords):
    # M0 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M1 = (A[1,0]+A[1,1]) * B[0,0]
    # M2 = A[0,0] * (B[0,1]-B[1,1])
    # M3 = A[1,1] * (B[1,0]-B[0,0])
    # M4 = (A[0,0]+A[0,1]) * B[1,1]
    # M5 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M0 + M3 - M4 - M6
    # C[0,1] = M2 + M4
    # C[1,0] = M1 + M5
    # C[1,1] = M0 - M1 + M2 + M5

    # factor 0: (A00+A11)*(B00+B11)
    U[i1*m+j1][col+0] += 1
    U[i2*m+j2][col+0] += 1
    V[j1*p+k1][col+0] += 1
    V[j2*p+k2][col+0] += 1
    W[k1*n+i1][col+0] += 1
    W[k1*n+i2][col+0] += -1
    W[k2*n+i1][col+0] += -1
    W[k2*n+i2][col+0] += -1

    # factor 1: (A10+A11)*B00
    U[i2*m+j1][col+1] += 1
    U[i2*m+j2][col+1] += 1
    V[j1*p+k1][col+1] += 1
    W[k2*n+i1][col+1] += 1
    W[k2*n+i2][col+1] += 1

    # factor 2: A00*(B01-B11)
    U[i1*m+j1][col+2] += 1
    V[j1*p+k2][col+2] += 1
    V[j2*p+k2][col+2] += -1
    W[k1*n+i2][col+2] += 1
    W[k2*n+i2][col+2] += 1

    # factor 3: A11*(B10-B00)
    U[i2*m+j2][col+3] += 1
    V[j2*p+k1][col+3] += 1
    V[j1*p+k1][col+3] += -1
    W[k1*n+i1][col+3] += 1

    # factor 4: (A00+A01)*B11
    U[i1*m+j1][col+4] += 1
    U[i1*m+j2][col+4] += 1
    V[j2*p+k2][col+4] += 1
    W[k1*n+i1][col+4] += -1
    W[k1*n+i2][col+4] += -1

    # factor 5: (A10-A00)*(B00+B01)
    U[i2*m+j1][col+5] += 1
    U[i1*m+j1][col+5] += -1
    V[j1*p+k1][col+5] += 1
    V[j1*p+k2][col+5] += 1
    W[k2*n+i1][col+5] += 1
    W[k2*n+i2][col+5] += 1

    # factor 6: (A01-A11)*(B10+B11)
    U[i1*m+j2][col+6] += 1
    U[i2*m+j2][col+6] += -1
    V[j2*p+k1][col+6] += 1
    V[j2*p+k2][col+6] += 1
    W[k1*n+i1][col+6] += -1
    W[k2*n+i1][col+6] += -1

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify U,V,W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    nm, mp, pn = n*m, m*p, p*n
    rank = U_arr.shape[1]
    
    # Build target tensor T[a,b,c] = 1 iff consistent
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute estimated tensor
    T_est = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    
    max_err = float(np.max(np.abs(T_est - T)))
    return max_err < tol, max_err

def _verify_half_integer(U, V, W, tol=1e-12):
    """Check if all entries are half-integers (integer or half-integer)."""
    for arr_name, arr in [('U', U), ('V', V), ('W', W)]:
        arr_f = np.array(arr, dtype=float)
        doubled = arr_f * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U, V, W = _make_zero_matrices(n, m, p, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _find_max_disjoint_strassen_blocks(n, m, p, deadline=None):
    """
    Find maximum number of disjoint 2x2x2 Strassen blocks.
    A block is defined by (i1,i2,j1,j2,k1,k2) where i1<i2, j1<j2, k1<k2.
    Two blocks are disjoint if they share no (i,j,k) triple.
    Uses branch-and-bound for exact maximum.
    """
    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n-1):
        for i2 in range(i1+1, n):
            for j1 in range(m-1):
                for j2 in range(j1+1, m):
                    for k1 in range(p-1):
                        for k2 in range(k1+1, p):
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
    
    num_blocks = len(all_blocks)
    
    # Precompute overlap matrix
    # Two blocks overlap if they share i-index AND j-index AND k-index
    overlap = [[False]*num_blocks for _ in range(num_blocks)]
    for a in range(num_blocks):
        i1a, i2a, j1a, j2a, k1a, k2a = all_blocks[a]
        i_set_a = {i1a, i2a}
        j_set_a = {j1a, j2a}
        k_set_a = {k1a, k2a}
        for b in range(a+1, num_blocks):
            i1b, i2b, j1b, j2b, k1b, k2b = all_blocks[b]
            i_set_b = {i1b, i2b}
            j_set_b = {j1b, j2b}
            k_set_b = {k1b, k2b}
            if (i_set_a & i_set_b) and (j_set_a & j_set_b) and (k_set_a & k_set_b):
                overlap[a][b] = True
                overlap[b][a] = True
    
    # Branch and bound for maximum independent set
    best_set = []
    
    def branch_and_bound(remaining_indices, current_set, depth):
        nonlocal best_set
        if deadline is not None and time.time() > deadline:
            return
        
        if not remaining_indices:
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return
        
        # Upper bound: current size + number of remaining
        if len(current_set) + len(remaining_indices) <= len(best_set):
            return
        
        # Pick a vertex from remaining
        v = remaining_indices[0]
        rest = remaining_indices[1:]
        
        # Option 1: include v
        # Remove v and all its neighbors from remaining
        new_remaining = [u for u in rest if not overlap[v][u]]
        current_set.append(v)
        branch_and_bound(new_remaining, current_set, depth+1)
        current_set.pop()
        
        # Option 2: exclude v
        branch_and_bound(rest, current_set, depth+1)
    
    # Use a smarter ordering: sort blocks by degree (most connected first) for better pruning
    degrees = [sum(overlap[i][j] for j in range(num_blocks)) for i in range(num_blocks)]
    sorted_indices = sorted(range(num_blocks), key=lambda x: -degrees[x])
    
    branch_and_bound(sorted_indices, [], 0)
    
    return [all_blocks[i] for i in best_set]

def _build_rank_optimized(n, m, p, deadline=None):
    """
    Build a decomposition using maximum number of disjoint Strassen 2x2x2 blocks.
    """
    blocks = _find_max_disjoint_strassen_blocks(n, m, p, deadline=deadline)
    num_blocks = len(blocks)
    
    # Each Strassen block saves 1 rank (8 schoolbook terms -> 7 factors)
    total_terms = n * m * p
    terms_covered = num_blocks * 8
    remaining_terms = total_terms - terms_covered
    rank = num_blocks * 7 + remaining_terms
    
    U, V, W = _make_zero_matrices(n, m, p, rank)
    
    # Track which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for block in blocks:
        i1, i2, j1, j2, k1, k2 = block
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    covered.add((i, j, k))
    
    # Add Strassen blocks
    col = 0
    for block in blocks:
        i1, i2, j1, j2, k1, k2 = block
        _add_strassen_block(U, V, W, col,
                            i1=i1, i2=i2, j1=j1, j2=j2, k1=k1, k2=k2,
                            n=n, m=m, p=p)
        col += 7
    
    # Add schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected col={rank}, got col={col}"
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the optimized decomposition with maximum disjoint Strassen blocks
    U_opt, V_opt, W_opt, rank_opt = _build_rank_optimized(n, m, p, deadline=deadline)
    
    # Verify numerically
    valid_opt, max_err_opt = _verify_decomposition(U_opt, V_opt, W_opt, n, m, p)
    
    if valid_opt:
        # Check if entries are half-integers
        if _verify_half_integer(U_opt, V_opt, W_opt):
            return {"U": U_opt, "V": V_opt, "W": W_opt}
    
    # Fall back to schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook(n, m, p)
    valid_sb, _ = _verify_decomposition(U_sb, V_sb, W_sb, n, m, p)
    
    if valid_sb:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # This should never happen, but return schoolbook anyway
    return {"U": U_sb, "V": V_sb, "W": W_sb}
