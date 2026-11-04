"""Strassen-based 4x5x6 matrix multiplication decomposition (rank 108)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, r_start):
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1

    r_offset = 0
    
    # 12 Strassen blocks for the 4x4x4 core
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                r_offset = add_strassen_block(i0, j0, k0, r_offset)
    
    # Strip: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6], rank 24
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1
            
    # Verify correctness quickly
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # Reshape for vectorized check
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Decomposition invalid at i={i}, j={j}, k={k}: sum={s}")
    
    return {"U": U, "V": V, "W": W}
