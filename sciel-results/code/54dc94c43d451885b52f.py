import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2 (rows of A), m=4 (inner), p=7 (cols of B)
    
    # Decomposition strategy:
    # Split m=4 into two blocks of 2: [0:2] and [2:4]
    # Split p=7 into [0:6] and [6:7]
    # For p[0:6]: split into three 2x2 blocks, each handled by Strassen (rank 7 each)
    # For p[6:7]: handled by schoolbook (rank 4 each)
    # Total per m-block: 3*7 + 4 = 25
    # Total rank: 2 * 25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 50)
    V = [[0] * rank for _ in range(m * p)]   # shape (28, 50)
    W = [[0] * rank for _ in range(p * n)]   # shape (14, 50)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += M2, C11 -= M2
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += M3, C11 += M3
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += M4, C10 += M4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= M5, C01 += M5
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += M6
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += M7
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
    
    def add_schoolbook_221(i0, j0, k0):
        """Schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
    
    # Process each m-block independently
    for j_start in [0, 2]:
        # Three 2x2 Strassen blocks for k = 0,2,4
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        # One schoolbook block for k = 6
        add_schoolbook_221(0, j_start, 6)
    
    # Verify we used exactly 'rank' columns
    assert col == rank, f"Expected {rank} columns, used {col}"
    
    # Quick numerical verification with random inputs
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reshape U, V, W to check: C[k*n+i, r] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # C[i,k] = sum_j T[i,j,k]
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # u_col has shape (n*m,), v_col has shape (m*p,), w_col has shape (p*n,)
        # T[i,j,k] = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        for i in range(n):
            for k in range(p):
                c_val = 0.0
                for j in range(m):
                    c_val += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                C_computed[i, k] += c_val
    
    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-10:
        # Fallback: return schoolbook if Strassen verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
