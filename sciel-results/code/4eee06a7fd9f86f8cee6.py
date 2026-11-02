import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Try rank-24 search first (if time permits)
    result_24 = _search_rank_24(n, m, p, deadline)
    if result_24 is not None:
        return result_24
    
    # Fallback to rank-28 Strassen
    return _strassen_rank_28(n, m, p)

def _strassen_rank_28(n, m, p):
    rank = 28
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n*m = 4, m*p = 16, p*n = 16
    
    # Strassen-based decomposition: 4 blocks of 2x2, each with 7 terms = 28 rank
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):  # 4 blocks
        k0 = 2 * b      # first column of this block
        k1 = 2 * b + 1  # second column of this block
        
        # M1 = (A00 + A11) * (B[0,k0] + B[1,k1])
        # U: A[0,0]=U[0], A[1,1]=U[3]
        U[0][col] += 1
        U[3][col] += 1
        # V: B[0,k0]=V[0*8+k0], B[1,k1]=V[1*8+k1]
        V[0 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # W: C[0,k0]+=M1, C[1,k1]+=M1
        # W[k*n+i, col]: C[0,k0] -> W[k0*2+0], C[1,k1] -> W[k1*2+1]
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B[0,k0]
        U[2][col] += 1  # A[1,0]
        U[3][col] += 1  # A[1,1]
        V[0 * p + k0][col] += 1
        # W: C[1,k0]+=M2, C[1,k1]-=M2
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] += -1
        col += 1
        
        # M3 = A00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1  # A[0,0]
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] += -1
        # W: C[0,k1]+=M3, C[1,k1]+=M3
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B[1,k0] - B[0,k0])
        U[3][col] += 1  # A[1,1]
        V[1 * p + k0][col] += 1
        V[0 * p + k0][col] += -1
        # W: C[0,k0]+=M4, C[1,k0]+=M4
        W[k0 * n + 0][col] += 1
        W[k0 * n + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B[1,k1]
        U[0][col] += 1  # A[0,0]
        U[1][col] += 1  # A[0,1]
        V[1 * p + k1][col] += 1
        # W: C[0,k0]-=M5, C[0,k1]+=M5
        W[k0 * n + 0][col] += -1
        W[k1 * n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B[0,k0] + B[0,k1])
        U[2][col] += 1  # A[1,0]
        U[0][col] += -1  # A[0,0]
        V[0 * p + k0][col] += 1
        V[0 * p + k1][col] += 1
        # W: C[1,k1]+=M6
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1  # A[0,1]
        U[3][col] += -1  # A[1,1]
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # W: C[0,k0]+=M7
        W[k0 * n + 0][col] += 1
        col += 1
    
        # Verify correctness
        try:
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
            T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for row_a in range(n):
                for col_a in range(m):
                    i = row_a * m + col_a
                    for row_b in range(m):
                        if row_b != col_a:
                            continue
                        for col_b in range(p):
                            j = row_b * p + col_b
                            k = col_b * n + row_a
                            T_expected[i, j, k] = 1
        
            if not np.allclose(T, T_expected, atol=1e-9):
                raise ValueError("Verification failed")
        except Exception as e:
            # If verification fails, fall back to schoolbook (rank 32)
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
        
            col = 0
            for i in range(n * m):
                for j in range(m * p):
                    a_row, a_col = divmod(i, m)
                    b_row, b_col = divmod(j, p)
                    if a_col == b_row:
                        k = b_col * n + a_row
                        U_school[i][col] = 1
                        V_school[j][col] = 1
                        W_school[k][col] = 1
                        col += 1
        
            return {"U": U_school, "V": V_school, "W": W_school}
    
        return {"U": U, "V": V, "W": W}

def _search_rank_24(n, m, p, deadline):
    """
    Search for rank-24 decomposition.
    Returns None if not found within time budget.
    """
    # For 2x2x8, it's unclear if rank 24 is achievable.
    # Strassen gives 28 (4 blocks x 7 terms).
    # Let's try a structured search: 4 blocks with 6 terms each.
    # This requires finding a 6-term decomposition for 2x2x2, which is known
    # to be impossible (minimum is 7). So rank 24 via this approach won't work.
    
    # Alternative: 3 blocks of 8 terms = 24. Each block handles 8/3 columns...
    # doesn't divide evenly.
    
    # Let's just return None and use the rank-28 fallback.
    # A more sophisticated search would be needed to beat 28.
    return None

def verify(U, V, W, n, m, p):
    """Verify the decomposition is correct"""
    import numpy as np
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Tensor T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # i in 0..nm-1, j in 0..mp-1, k in 0..pn-1
    # U shape (nm, rank), V shape (mp, rank), W shape (pn, rank)
    
    # Construct T
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Construct Expected T
    # A is n x m, B is m x p, C is n x p
    # T[i, j, k] is 1 if C_{row_c, col_c} = A_{row_a, col_a} * B_{row_b, col_b}
    # i = row_a * m + col_a
    # j = row_b * p + col_b
    # k = col_c * n + row_c  (Based on W index convention: k*n+i where k is col, i is row)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                if row_b != col_a: continue
                for col_b in range(p):
                    j = row_b * p + col_b
                    row_c = row_a
                    col_c = col_b
                    k = col_c * n + row_c
                    T_expected[i, j, k] = 1
                    
    if not np.allclose(T, T_expected):
        raise ValueError("Verification failed")
