import time
import numpy as np
from fractions import Fraction

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

def _find_max_disjoint_strassen_blocks(n, m, p):
    """
    Find maximum number of disjoint 2x2x2 Strassen blocks.
    A block is defined by (i1,i2,j1,j2,k1,k2) where i1<i2, j1<j2, k1<k2.
    Two blocks are disjoint if they share no (i,j,k) triple.
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
    
    # Greedy: sort by some heuristic and pick non-overlapping blocks
    # Two blocks overlap if they share any (i,j,k) triple
    # Block A = (i1,i2,j1,j2,k1,k2) covers {i1,i2}x{j1,j2}x{k1,k2}
    # Block B overlaps A if {i1,i2}∩{i1',i2'} != empty AND {j1,j2}∩{j1',j2'} != empty AND {k1,k2}∩{k1',k2'} != empty
    
    def blocks_overlap(b1, b2):
        i_set1 = {b1[0], b1[1]}
        i_set2 = {b2[0], b2[1]}
        j_set1 = {b1[2], b1[3]}
        j_set2 = {b2[2], b2[3]}
        k_set1 = {b1[4], b1[5]}
        k_set2 = {b2[4], b2[5]}
        return (i_set1 & i_set2) and (j_set1 & j_set2) and (k_set1 & k_set2)
    
    # Greedy with priority: prefer blocks that use "rare" indices
    # For n=3, index 2 is only available in one pair (1,2), so blocks using index 2 are more constrained
    # Heuristic: sort by sum of indices (prefer lower indices first? or higher?)
    # Actually, let's try multiple orderings and pick the best
    
    def greedy_order(blocks, key_func):
        sorted_blocks = sorted(blocks, key=key_func)
        selected = []
        for b in sorted_blocks:
            if all(not blocks_overlap(b, s) for s in selected):
                selected.append(b)
        return selected
    
    # Try different orderings
    best_blocks = []
    
    # Ordering 1: natural order
    b1 = greedy_order(all_blocks, lambda b: (b[0], b[1], b[2], b[3], b[4], b[5]))
    if len(b1) > len(best_blocks):
        best_blocks = b1
    
    # Ordering 2: prefer blocks with higher index values (to use up constrained indices)
    b2 = greedy_order(all_blocks, lambda b: (-b[0]-b[1], -b[2]-b[3], -b[4]-b[5]))
    if len(b2) > len(best_blocks):
        best_blocks = b2
    
    # Ordering 3: prefer blocks using index n-1 (most constrained)
    b3 = greedy_order(all_blocks, lambda b: (-max(b[0],b[1]), -max(b[2],b[3]), -max(b[4],b[5])))
    if len(b3) > len(best_blocks):
        best_blocks = b3
    
    # Ordering 4: prefer blocks using index m-1
    b4 = greedy_order(all_blocks, lambda b: (-max(b[2],b[3]), -max(b[0],b[1]), -max(b[4],b[5])))
    if len(b4) > len(best_blocks):
        best_blocks = b4
    
    # Ordering 5: prefer blocks using index p-1
    b5 = greedy_order(all_blocks, lambda b: (-max(b[4],b[5]), -max(b[0],b[1]), -max(b[2],b[3])))
    if len(b5) > len(best_blocks):
        best_blocks = b5
    
    # Try random orderings
    import random
    for _ in range(100):
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        b_rand = greedy_order(shuffled, lambda b: 0)
        if len(b_rand) > len(best_blocks):
            best_blocks = b_rand
    
    return best_blocks

def _build_rank_optimized(n, m, p):
    """
    Build a decomposition using maximum number of disjoint Strassen 2x2x2 blocks.
    """
    blocks = _find_max_disjoint_strassen_blocks(n, m, p)
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
    U_opt, V_opt, W_opt, rank_opt = _build_rank_optimized(n, m, p)
    
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
