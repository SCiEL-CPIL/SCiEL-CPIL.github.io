import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_start_time = time.time()
TIME_LIMIT = 145.0

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    # Compute decomposition
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.allclose(T_decomp, T_target, atol=1e-9)

def _schoolbook_decomposition():
    """Pure schoolbook decomposition, rank = n*m*p = 2*3*9 = 54"""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(U, V, W, col, u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11):
    """Add Strassen 2x2 block at given column offset"""
    # M1 = (a+d)(e+h)
    U[u_a][col] += 1
    U[u_d][col] += 1
    V[v_e][col] += 1
    V[v_h][col] += 1
    
    # M2 = (c+d)e
    U[u_c][col+1] += 1
    U[u_d][col+1] += 1
    V[v_e][col+1] += 1
    
    # M3 = a(f-h)
    U[u_a][col+2] += 1
    V[v_f][col+2] += 1
    V[v_h][col+2] -= 1
    
    # M4 = d(g-e)
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    V[v_e][col+3] -= 1
    
    # M5 = (a+b)h
    U[u_a][col+4] += 1
    U[u_b][col+4] += 1
    V[v_h][col+4] += 1
    
    # M6 = (c-a)(e+f)
    U[u_c][col+5] += 1
    U[u_a][col+5] -= 1
    V[v_e][col+5] += 1
    V[v_f][col+5] += 1
    
    # M7 = (b-d)(g+h)
    U[u_b][col+6] += 1
    U[u_d][col+6] -= 1
    V[v_g][col+6] += 1
    V[v_h][col+6] += 1
    
    # C00 = M1 + M4 - M5 + M7
    W[w_00][col] += 1
    W[w_00][col+3] += 1
    W[w_00][col+4] -= 1
    W[w_00][col+6] += 1
    
    # C01 = M3 + M5
    W[w_01][col+2] += 1
    W[w_01][col+4] += 1
    
    # C10 = M2 + M4
    W[w_10][col+1] += 1
    W[w_10][col+3] += 1
    
    # C11 = M1 - M2 + M3 + M6
    W[w_11][col] += 1
    W[w_11][col+1] -= 1
    W[w_11][col+2] += 1
    W[w_11][col+5] += 1

def _rank50_decomposition():
    """Strassen-based rank 50 decomposition"""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        _strassen_2x2_block(U, V, W, col, u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11)
        col += 7
    
    # 1 block of 2x2 @ 2x1
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # A_R @ B_B: rank 18
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_by_nullspace(U, V, W, n, m, p):
    """Try to reduce rank by finding linear dependencies in the factor matrices"""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check for zero columns
    for r in range(rank):
        if np.allclose(U_arr[:, r], 0) and np.allclose(V_arr[:, r], 0) and np.allclose(W_arr[:, r], 0):
            U_arr = np.delete(U_arr, r, axis=1)
            V_arr = np.delete(V_arr, r, axis=1)
            W_arr = np.delete(W_arr, r, axis=1)
            rank -= 1
            r -= 1
    
    return U_arr, V_arr, W_arr, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with rank 50 Strassen decomposition
    result = _rank50_decomposition()
    
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_by_nullspace(U, V, W, n, m, p)
    
    if rank < 50:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return result
