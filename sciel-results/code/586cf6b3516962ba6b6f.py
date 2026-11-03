import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _strassen_2x2():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x3():
    """
    Schoolbook decomposition for 2x2x3 matrix multiplication.
    U shape: (4, 12), V shape: (6, 12), W shape: (6, 12)
    Indices: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 3
    rank = n * m * p  # 12
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_1x2x3():
    """
    Schoolbook decomposition for 1x2x3 matrix multiplication.
    U shape: (2, 6), V shape: (6, 6), W shape: (3, 6)
    Indices: U[i*2+j, r], V[j*3+k, r], W[k*1+i, r]
    """
    n, m, p = 1, 2, 3
    rank = n * m * p  # 6
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _schoolbook_1x2x2():
    """
    Schoolbook decomposition for 1x2x2 matrix multiplication.
    U shape: (2, 4), V shape: (4, 4), W shape: (2, 4)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*1+i, r]
    """
    n, m, p = 1, 2, 2
    rank = n * m * p  # 4
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _map_and_append(U_list, V_list, W_list, U_block, V_block, W_block, 
                     i_off, j_off, k_off, i_size, j_size, k_size, 
                     n_global, m_global, p_global):
    """
    Map a local block decomposition to global indices and append to lists.
    """
    rank = U_block.shape[1]
    U_g = np.zeros((n_global * m_global, rank), dtype=int)
    V_g = np.zeros((m_global * p_global, rank), dtype=int)
    W_g = np.zeros((p_global * n_global, rank), dtype=int)
    
    for r in range(rank):
        # Map U: local index li*j_size + lj -> global (i_off+li)*m_global + (j_off+lj)
        for li in range(i_size):
            for lj in range(j_size):
                u_val = U_block[li * j_size + lj, r]
                if u_val != 0:
                    gi = i_off + li
                    gj = j_off + lj
                    U_g[gi * m_global + gj, r] = u_val
        
        # Map V: local index lj*k_size + lk -> global (j_off+lj)*p_global + (k_off+lk)
        for lj in range(j_size):
            for lk in range(k_size):
                v_val = V_block[lj * k_size + lk, r]
                if v_val != 0:
                    gj = j_off + lj
                    gk = k_off + lk
                    V_g[gj * p_global + gk, r] = v_val
        
        # Map W: local index lk*i_size + li -> global (k_off+lk)*n_global + (i_off+li)
        for lk in range(k_size):
            for li in range(i_size):
                w_val = W_block[lk * i_size + li, r]
                if w_val != 0:
                    gk = k_off + lk
                    gi = i_off + li
                    W_g[gk * n_global + gi, r] = w_val
    
    U_list.append(U_g)
    V_list.append(V_g)
    W_list.append(W_g)

def _construct_block_decomposition():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block decomposition:
    # i: {0,1} (size 2) and {2} (size 1)
    # j: {0,1} (size 2) and {2,3} (size 2)
    # k: {0,1,2} (size 3) and {3,4} (size 2)
    
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1,2} -> 2x2x3, rank 12
    U1, V1, W1 = _schoolbook_2x2x3()
    _map_and_append(U_list, V_list, W_list, U1, V1, W1, 
                    0, 0, 0, 2, 2, 3, n, m, p)
    
    # Block 2: i∈{0,1}, j∈{2,3}, k∈{0,1,2} -> 2x2x3, rank 12
    U2, V2, W2 = _schoolbook_2x2x3()
    _map_and_append(U_list, V_list, W_list, U2, V2, W2, 
                    0, 2, 0, 2, 2, 3, n, m, p)
    
    # Block 3: i∈{0,1}, j∈{0,1}, k∈{3,4} -> 2x2x2, rank 7 (Strassen)
    U3, V3, W3 = _strassen_2x2()
    _map_and_append(U_list, V_list, W_list, U3, V3, W3, 
                    0, 0, 3, 2, 2, 2, n, m, p)
    
    # Block 4: i∈{0,1}, j∈{2,3}, k∈{3,4} -> 2x2x2, rank 7 (Strassen)
    U4, V4, W4 = _strassen_2x2()
    _map_and_append(U_list, V_list, W_list, U4, V4, W4, 
                    0, 2, 3, 2, 2, 2, n, m, p)
    
    # Block 5: i∈{2}, j∈{0,1}, k∈{0,1,2} -> 1x2x3, rank 6
    U5, V5, W5 = _schoolbook_1x2x3()
    _map_and_append(U_list, V_list, W_list, U5, V5, W5, 
                    2, 0, 0, 1, 2, 3, n, m, p)
    
    # Block 6: i∈{2}, j∈{2,3}, k∈{0,1,2} -> 1x2x3, rank 6
    U6, V6, W6 = _schoolbook_1x2x3()
    _map_and_append(U_list, V_list, W_list, U6, V6, W6, 
                    2, 2, 0, 1, 2, 3, n, m, p)
    
    # Block 7: i∈{2}, j∈{0,1}, k∈{3,4} -> 1x2x2, rank 4
    U7, V7, W7 = _schoolbook_1x2x2()
    _map_and_append(U_list, V_list, W_list, U7, V7, W7, 
                    2, 0, 3, 1, 2, 2, n, m, p)
    
    # Block 8: i∈{2}, j∈{2,3}, k∈{3,4} -> 1x2x2, rank 4
    U8, V8, W8 = _schoolbook_1x2x2()
    _map_and_append(U_list, V_list, W_list, U8, V8, W8, 
                    2, 2, 3, 1, 2, 2, n, m, p)
    
    U_final = np.hstack(U_list)
    V_final = np.hstack(V_list)
    W_final = np.hstack(W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try block decomposition
    result = _construct_block_decomposition()
    
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
