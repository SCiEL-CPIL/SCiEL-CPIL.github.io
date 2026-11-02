import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0

def _check_deadline():
    if time.time() > _DEADLINE - 5.0:
        return True
    return False

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T[a, b, d] = 1.0
    return T

def _verify(n, m, p, U, V, W):
    T = _build_tensor(n, m, p)
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T, atol=1e-10, rtol=0)

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 with rank 7 (Strassen).
    A, B, C flattened row-major: [00, 01, 10, 11]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x3x2_rank11():
    """
    Rank-11 decomposition for 2x3 * 3x2 = 2x2.
    A: 2x3, a = i*3+j
    B: 3x2, b = j*2+k
    C: 2x2, d = k*2+i
    """
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Strassen A indices [0,1,2,3] map to A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    a_map = [0, 1, 3, 4]
    # Strassen B indices [0,1,2,3] map to B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    b_map = [0, 1, 2, 3]
    # Strassen C indices [0,1,2,3] map to C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    c_map = [0, 1, 2, 3]
    
    for c in range(7):
        for s_a, val in enumerate(U_s[:, c]):
            if val != 0:
                U[a_map[s_a], c] += val
        for s_b, val in enumerate(V_s[:, c]):
            if val != 0:
                V[b_map[s_b], c] += val
        for s_c, val in enumerate(W_s[:, c]):
            if val != 0:
                W[c_map[s_c], c] += val
    
    # j=2 contribution: C[i,k] += A[i,2]*B[2,k]
    for ii in range(2):
        for kk in range(2):
            c = 7 + ii * 2 + kk
            U[ii * 3 + 2, c] = 1
            V[2 * 2 + kk, c] = 1
            W[kk * 2 + ii, c] = 1
    
    return U, V, W

def _build_2x3x8_rank44():
    """
    Rank-44 decomposition for 2x3 * 3x8 = 2x8.
    Split p=8 into 4 blocks of size 2, each rank-11.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    num_blocks = p // block_size
    r_sub = 11
    total_rank = num_blocks * r_sub
    
    U_sub, V_sub, W_sub = _build_2x3x2_rank11()
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for b in range(num_blocks):
        k_offset = b * block_size
        col_offset = b * r_sub
        
        U_full[:, col_offset:col_offset+r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(block_size):
                b_local = j * block_size + k_prime
                b_global = j * p + k_offset + k_prime
                V_full[b_global, col_offset:col_offset+r_sub] = V_sub[b_local]
        
        for k_prime in range(block_size):
            for i in range(n):
                d_local = k_prime * n + i
                d_global = (k_offset + k_prime) * n + i
                W_full[d_global, col_offset:col_offset+r_sub] = W_sub[d_local]
    
    return U_full, V_full, W_full

def construct():
    n, m, p = 2, 3, 8
    
    if not _check_deadline():
        U44, V44, W44 = _build_2x3x8_rank44()
        if _verify(n, m, p, U44, V44, W44):
            return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
    
    # Fallback to schoolbook rank 48
    rank_school = n * m * p
    U_fb = np.zeros((n*m, rank_school))
    V_fb = np.zeros((m*p, rank_school))
    W_fb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
