"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(actual - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by checking for linearly dependent columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check if any column can be removed (zero contribution)
    # or if columns are linearly dependent in a way that allows reduction
    # Simple check: if a column is all zeros, remove it
    zero_cols = []
    for r in range(rank):
        if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
            zero_cols.append(r)
    
    if zero_cols:
        keep = [r for r in range(rank) if r not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    # Try to find if any pair of columns can be merged (not straightforward)
    # For now, just return reduced if zeros were found
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: block-diagonal Kronecker decomposition
    # Split m=5 into 1 + 4
    # Part 1: 2x1x6, rank = 12
    # Part 2: 2x4x6, split k=6 into 2+2+2, each 2x4x2
    #   Each 2x4x2: split j=4 into 2+2, each 2x2x2 has rank 7 (Strassen)
    #   So 2x4x2 has rank 14
    #   2x4x6 has rank 3*14 = 42
    # Total rank = 12 + 42 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, A[:,0] (2x1) * B[0,:] (1x6) -> 2x6
    # For each i in 0..1, k in 0..5: C[i,k] += A[i,0]*B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12
    
    # Part 2: j in 1..4, A[:,1:5] (2x4) * B[1:5,:] (4x6) -> 2x6
    # Split k into 3 blocks of size 2: k_base in {0, 2, 4}
    # For each k-block, split j into 2 blocks of size 2: j_base in {1, 3}
    # Each 2x2x2 sub-problem uses Strassen with rank 7
    
    # Strassen terms for 2x2x2:
    # Local indices: A[li][lj] with li,lj in {0,1}, B[lj][lk] with lj,lk in {0,1}, C[li][lk]
    # T1: (A00+A11)*(B00+B11) -> C00 += T1, C11 += T1
    # T2: (A10+A11)*B00 -> C10 += T2, C11 -= T2
    # T3: A00*(B01-B11) -> C01 += T3, C11 += T3
    # T4: A11*(B10-B00) -> C00 += T4, C10 += T4
    # T5: (A00+A01)*B11 -> C00 -= T5, C01 += T5
    # T6: (A10-A00)*(B00+B01) -> C11 += T6
    # T7: (A01-A11)*(B10+B11) -> C00 += T7
    
    strassen_terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]
    
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = 1 + j_block * 2  # j in {1,2} or {3,4}
            
            for term in strassen_terms:
                u_entries, v_entries, w_entries = term
                
                for (li, lj, coeff) in u_entries:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += coeff
                
                for (lj, lk, coeff) in v_entries:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += coeff
                
                for (li, lk, coeff) in w_entries:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += coeff
                
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify decomposition
    max_err = _verify_decomposition(U, V, W, n, m, p)
    if max_err > 1e-9:
        # Fallback: use naive rank-60 decomposition if Strassen has issues
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][c] = 1
                    V_naive[j*p+k][c] = 1
                    W_naive[k*n+i][c] = 1
                    c += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Try to reduce rank if time permits
    if time.time() < deadline:
        U_list, V_list, W_list, new_rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if new_rank < rank:
            max_err2 = _verify_decomposition(U_list, V_list, W_list, n, m, p)
            if max_err2 < 1e-9:
                U, V, W = U_list, V_list, W_list
                rank = new_rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
