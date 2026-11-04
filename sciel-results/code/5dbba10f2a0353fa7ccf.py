import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Winograd-style rank-14 decomposition for 2x3x3
    # A is 2x3, B is 3x3, C is 2x3
    # A indices: a00=0, a01=1, a02=2, a10=3, a11=4, a12=5
    # B indices: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
    
    U_block = np.zeros((6, 14))
    V_block = np.zeros((9, 14))
    W_block = np.zeros((6, 14))
    
    # M0 = (a00+a10)(b00+b10)
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    
    # M1 = (a00-a10)(b00-b10)
    U_block[0, 1] = 1; U_block[3, 1] = -1
    V_block[0, 1] = 1; V_block[3, 1] = -1
    
    # M2 = (a00+a02)(b00+b02)
    U_block[0, 2] = 1; U_block[2, 2] = 1
    V_block[0, 2] = 1; V_block[2, 2] = 1
    
    # M3 = (a00-a02)(b00-b02)
    U_block[0, 3] = 1; U_block[2, 3] = -1
    V_block[0, 3] = 1; V_block[2, 3] = -1
    
    # M4 = (a02+a12)(b02+b12)
    U_block[2, 4] = 1; U_block[5, 4] = 1
    V_block[2, 4] = 1; V_block[5, 4] = 1
    
    # M5 = (a02-a12)(b02-b12)
    U_block[2, 5] = 1; U_block[5, 5] = -1
    V_block[2, 5] = 1; V_block[5, 5] = -1
    
    # M6 = (a10+a12)(b10+b12)
    U_block[3, 6] = 1; U_block[5, 6] = 1
    V_block[3, 6] = 1; V_block[5, 6] = 1
    
    # M7 = (a10-a12)(b10-b12)
    U_block[3, 7] = 1; U_block[5, 7] = -1
    V_block[3, 7] = 1; V_block[5, 7] = -1
    
    # M8 = a01*b01
    U_block[1, 8] = 1
    V_block[1, 8] = 1
    
    # M9 = a01*b11
    U_block[1, 9] = 1
    V_block[4, 9] = 1
    
    # M10 = a01*b21
    U_block[1, 10] = 1
    V_block[7, 10] = 1
    
    # M11 = a11*b11
    U_block[4, 11] = 1
    V_block[4, 11] = 1
    
    # M12 = a11*b21
    U_block[4, 12] = 1
    V_block[7, 12] = 1
    
    # M13 = a12*b22
    U_block[5, 13] = 1
    V_block[8, 13] = 1
    
    # Now determine W_block
    # C = A @ B where C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W index convention: W[k*n+i] where n=2, so W[2*k+i]
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    
    # Let's compute W by solving the linear system
    # For each output C[i,k], express it in terms of M0...M13
    
    # C[0,0] = (1/4)[(M0+M1) + (M2+M3) - (M4+M5) - (M6+M7)] + a01*b10 + a02*b20
    # Wait, this approach is getting complicated. Let me use the numerical approach.
    
    rng = np.random.RandomState(42)
    num_samples = 50
    M_matrix = np.zeros((num_samples, 14))
    C_matrix = np.zeros((num_samples, 6))
    
    for s in range(num_samples):
        A_s = rng.rand(6)
        B_s = rng.rand(9)
        
        for r in range(14):
            u_val = np.dot(U_block[:, r], A_s)
            v_val = np.dot(V_block[:, r], B_s)
            M_matrix[s, r] = u_val * v_val
        
        A_mat = A_s.reshape(2, 3)
        B_mat = B_s.reshape(3, 3)
        C_direct = A_mat @ B_mat
        
        C_matrix[s, 0] = C_direct[0, 0]
        C_matrix[s, 1] = C_direct[1, 0]
        C_matrix[s, 2] = C_direct[0, 1]
        C_matrix[s, 3] = C_direct[1, 1]
        C_matrix[s, 4] = C_direct[0, 2]
        C_matrix[s, 5] = C_direct[1, 2]
    
    W_solve, residuals, rank_solve, sv = np.linalg.lstsq(M_matrix, C_matrix, rcond=None)
    W_block = W_solve.T
    
    if not np.allclose(M_matrix @ W_block.T, C_matrix, atol=1e-8):
        return _fallback_schoolbook(2, 3, 9)
    
    W_rounded = np.round(W_block * 2) / 2
    if np.allclose(W_block, W_rounded, atol=1e-6):
        W_block = W_rounded
    
    # Build full 2x3x9 decomposition with rank 42
    U_full = np.zeros((6, 42))
    V_full = np.zeros((27, 42))
    W_full = np.zeros((18, 42))
    
    for block in range(3):
        U_full[:, block*14:(block+1)*14] = U_block
        
        for j in range(3):
            for k_local in range(3):
                k = 3 * block + k_local
                V_full[j * 9 + k, block*14:(block+1)*14] = V_block[j * 3 + k_local, :]
        
        for k_local in range(3):
            k = 3 * block + k_local
            for i in range(2):
                w_idx = k * 2 + i
                local_w_idx = k_local * 2 + i
                W_full[w_idx, block*14:(block+1)*14] = W_block[local_w_idx, :]
    
    # Verify
    T = np.zeros((18, 6, 27))
    for i in range(2):
        for j in range(3):
            for k in range(9):
                T[k * 2 + i, i * 3 + j, j * 9 + k] = 1.0
    
    T_decomp = np.zeros_like(T)
    for r in range(42):
        T_decomp += np.einsum('i,j,k->ijk', U_full[:, r], V_full[:, r], W_full[:, r])
    
    if np.allclose(T_decomp, T, atol=1e-10):
        U_list = [[float(U_full[i, r]) for r in range(42)] for i in range(6)]
        V_list = [[float(V_full[i, r]) for r in range(42)] for i in range(27)]
        W_list = [[float(W_full[i, r]) for r in range(42)] for i in range(18)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return _fallback_schoolbook(2, 3, 9)

def _fallback_schoolbook(n, m, p):
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
