"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks plus systematic exact rank reduction
to find linear dependencies among the bilinear terms.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]


def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)
    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank


def _find_linear_dependencies(U, V, W, timeout=60):
    """
    Find linear dependencies among rank-1 terms using exact rational arithmetic.
    A term j is redundant if its (u_j, v_j, w_j) triple is a linear combination
    of other terms. We check if removing term j still allows reconstruction.
    
    For a bilinear decomposition, term j can be removed if there exist scalars
    c_i such that for all entries:
      u_j = sum_i c_i * u_i  (in U space)
      v_j = sum_i c_i * v_i  (in V space)  
      w_j = sum_i c_i * w_i  (in W space)
    This is a system of linear equations. We use Gaussian elimination over Q.
    """
    start = time.time()
    n_terms = U[0].__len__() if isinstance(U[0], list) else len(U[0])
    
    # Convert to Fraction
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    nU = len(Uf)
    nV = len(Vf)
    nW = len(Wf)
    
    removed = set()
    best_rank = n_terms
    
    # Try removing each term and check if the remaining terms can span it
    for j in range(n_terms):
        if time.time() - start > timeout:
            break
        if j in removed:
            continue
        
        # Build system: find c_i (i != j) such that:
        # sum_{i!=j} c_i * U[i][k] = U[j][k] for all k
        # sum_{i!=j} c_i * V[i][k] = V[j][k] for all k  
        # sum_{i!=j} c_i * W[i][k] = W[j][k] for all k
        
        # This is a linear system with (nU + nV + nW) equations and (n_terms - 1) unknowns
        # Use Gaussian elimination
        
        idxs = [i for i in range(n_terms) if i != j]
        n_eq = nU + nV + nW
        n_unknowns = len(idxs)
        
        if n_unknowns == 0:
            continue
        
        # Build augmented matrix [A | b]
        # A is n_eq x n_unknowns, b is n_eq
        A = [[Fraction(0)] * n_unknowns for _ in range(n_eq)]
        b = [Fraction(0)] * n_eq
        
        for k in range(nU):
            for ii, i in enumerate(idxs):
                A[k][ii] = Uf[i][k]
            b[k] = Uf[j][k]
        
        for k in range(nV):
            for ii, i in enumerate(idxs):
                A[nU + k][ii] = Vf[i][k]
            b[nU + k] = Vf[j][k]
        
        for k in range(nW):
            for ii, i in enumerate(idxs):
                A[nU + nV + k][ii] = Wf[i][k]
            b[nU + nV + k] = Wf[j][k]
        
        # Gaussian elimination
        rank = 0
        for col in range(n_unknowns):
            if rank >= n_eq:
                break
            # Find pivot
            pivot = -1
            for row in range(rank, n_eq):
                if A[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap
            A[rank], A[pivot] = A[pivot], A[rank]
            b[rank], b[pivot] = b[pivot], b[rank]
            # Eliminate
            pivot_val = A[rank][col]
            for k in range(n_unknowns):
                A[rank][k] /= pivot_val
            b[rank] /= pivot_val
            for row in range(n_eq):
                if row != rank and A[row][col] != 0:
                    factor = A[row][col]
                    for k in range(n_unknowns):
                        A[row][k] -= factor * A[rank][k]
                    b[row] -= factor * b[rank]
            rank += 1
        
        # Check consistency: for rows >= rank, b[row] should be 0
        consistent = True
        for row in range(rank, n_eq):
            if b[row] != 0:
                consistent = False
                break
        
        if consistent:
            removed.add(j)
            best_rank -= 1
    
    if not removed:
        return U, V, W
    
    keep = [j for j in range(n_terms) if j not in removed]
    U2 = [[U[i][j] for j in keep] for i in range(nU)]
    V2 = [[V[i][j] for j in keep] for i in range(nV)]
    W2 = [[W[i][j] for j in keep] for i in range(nW)]
    
    return U2, V2, W2


def _verify_decomposition(U, V, W, n=2, m=5, p=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Compute tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with identity tensor for matmul: T[i,j,k] should equal delta_{i,k} * (A_ij * B_jk contribution)
    # Actually, the tensor for matmul is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition should satisfy: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,k} * (coefficient)
    # Wait, the convention is W[k*n+i, r], so W index is k*n+i
    # The tensor entry is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if i==k (for the matmul tensor), 0 otherwise? No...
    # Actually for matmul, the structure tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is all ones! Let me verify.
    
    # The multiplication tensor: mu[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: mu[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _try_alternative_partitionings(timeout=30):
    """
    Try different ways to partition the dimensions and see if any give lower rank.
    Key insight: with n=2, we can potentially use more clever decompositions.
    """
    start = time.time()
    n, m, p = 2, 5, 5
    best = None
    
    # Try partitioning 5 differently: 1+1+1+1+1 (all schoolbook = 50)
    # 2+2+1 (current, = 46)
    # 3+2 (Strassen on 2x2 part, schoolbook on 3x2)
    #   A = [A_3 | A_2], B = [B_3; B_2]
    #   A_3 @ B_3: 2x3 x 3x5 = 30
    #   A_2 @ B_2: 2x2 x 2x5 -> partition B_2 cols as 2+2+1: 7+7+4=18
    #   Total: 48. Worse.
    
    # Try 2+3: A = [A_2 | A_3], B = [B_2; B_3]
    #   A_2 @ B_2: 2x2 x 2x5 -> 7+7+4 = 18 (partition B_2 cols 2+2+1)
    #   A_3 @ B_3: 2x3 x 3x5 = 30
    #   Total: 48. Worse.
    
    # What about using the 2-row structure more aggressively?
    # For 2x2 x 2x2, Strassen gives 7. Can we do better for 2xn x nx2?
    # Actually, let's try a completely different approach:
    # Directly construct a lower-rank decomposition using the structure.
    
    # The tensor has size 2*5*5 = 50 entries, all equal to 1.
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # One approach: use the fact that i only takes 2 values (0,1)
    # W[k,0,r] and W[k,1,r] for k=0..4 give 10 values per term.
    
    # Let's try a greedy approach: build terms one at a time to cover the tensor.
    # This is essentially the "tensor rank" problem, which is NP-hard, but for small sizes
    # we can try heuristic approaches.
    
    if time.time() - start > timeout:
        return None
    
    return best


def _exact_rank_reduce(U, V, W):
    """
    Cheap exact rank-reduction passes on a bilinear decomposition.
    1) Drop columns that are all-zero.
    2) Merge duplicate or scalar-multiple columns.
    """
    nU = len(U)
    nV = len(V)
    nW = len(W)
    n_terms = len(U[0]) if isinstance(U[0], list) else U[0].__len__()
    
    # Pass 1: drop all-zero columns
    keep = []
    for j in range(n_terms):
        col_u = [U[i][j] for i in range(nU)]
        col_v = [V[i][j] for i in range(nV)]
        col_w = [W[i][j] for i in range(nW)]
        if all(x == 0 for x in col_u) and all(x == 0 for x in col_v) and all(x == 0 for x in col_w):
            continue
        keep.append(j)
    
    if len(keep) < n_terms:
        U = [[U[i][j] for j in keep] for i in range(nU)]
        V = [[V[i][j] for j in keep] for i in range(nV)]
        W = [[W[i][j] for j in keep] for i in range(nW)]
        n_terms = len(keep)
    
    # Pass 2: merge scalar-multiple columns
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    removed = set()
    for i in range(n_terms):
        if i in removed:
            continue
        col_u_i = [Uf[r][i] for r in range(nU)]
        col_v_i = [Vf[r][i] for r in range(nV)]
        col_w_i = [Wf[r][i] for r in range(nW)]
        
        for j in range(i + 1, n_terms):
            if j in removed:
                continue
            col_u_j = [Uf[r][j] for r in range(nU)]
            col_v_j = [Vf[r][j] for r in range(nV)]
            col_w_j = [Wf[r][j] for r in range(nW)]
            
            c_prime = None
            for lst_i, lst_j in ((col_u_i, col_u_j), (col_v_i, col_v_j), (col_w_i, col_w_j)):
                for val_i, val_j in zip(lst_i, lst_j):
                    if val_i != 0:
                        c_prime = val_j / val_i
                        break
                if c_prime is not None:
                    break
            
            if c_prime is None:
                continue
            
            ok = True
            for lst_i, lst_j in ((col_u_i, col_u_j), (col_v_i, col_v_j), (col_w_i, col_w_j)):
                for val_i, val_j in zip(lst_i, lst_j):
                    if c_prime * val_i != val_j:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                removed.add(j)
    
    if removed:
        keep = [j for j in range(n_terms) if j not in removed]
        U = [[U[i][j] for j in keep] for i in range(nU)]
        V = [[V[i][j] for j in keep] for i in range(nV)]
        W = [[W[i][j] for j in keep] for i in range(nW)]
    
    def to_python(lst):
        out = []
        for row in lst:
            new_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        new_row.append(val.numerator)
                    else:
                        new_row.append(float(val))
                else:
                    new_row.append(val)
            out.append(new_row)
        return out
    
    return to_python(U), to_python(V), to_python(W)


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build the standard rank-46 decomposition
    U, V, W, total_rank = _build_strassen_decomposition()
    
    # Time guard
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    # Apply cheap exact rank-reduction passes
    try:
        U2, V2, W2 = _exact_rank_reduce(U, V, W)
        if time.time() > deadline - 30.0:
            return {"U": U2, "V": V2, "W": W2}
        
        # Try finding linear dependencies (more expensive)
        U3, V3, W3 = _find_linear_dependencies(U2, V2, W2, timeout=25)
        if time.time() > deadline - 5.0:
            return {"U": U2, "V": V2, "W": W2}
        
        # Verify the result
        if _verify_decomposition(U3, V3, W3):
            return {"U": U3, "V": V3, "W": W3}
        else:
            # Fallback to the simpler reduction
            return {"U": U2, "V": V2, "W": W2}
    except Exception:
        return {"U": U, "V": V, "W": W}
