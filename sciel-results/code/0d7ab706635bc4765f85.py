import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 4
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 multiplication (local indices 0..3 = [00,01,10,11]):
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y01-Y11)
    # M4: X11(Y10-Y00)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7, Z01 = M3+M5, Z10 = M2+M4, Z11 = M1-M2+M3+M6
    
    # For C_left = A11*B11 + A12*B21:
    # Shared factors: (A11+A12) and (A11-A12)
    # 10 multiplications instead of 14:
    # L1: (A11+A12)00 * (B11+B21)00
    # L2: (A11+A12)11 * (B11+B21)11
    # L3: (A11+A12)(00+11) * (B11+B21)(00+11)
    # L4: (A11+A12)10 * B11 00
    # L5: (A11+A12)10 * B21 00
    # L6: (A11+A12)00 * (B11 01 - B21 11)
    # L7: (A11+A12)11 * (B21 10 - B11 00)
    # L8: (A11+A12)(00+01) * B11 11
    # L9: (A11+A12)(00+01) * B21 11
    # L10: (A11-A12)00 * (B11+B21)(00+01)
    
    # For C_right = A11*B12 + A12*B22:
    # 10 multiplications:
    # R1: (A11+A12)00 * (B12+B22)00
    # R2: (A11+A12)11 * (B12+B22)11
    # R3: (A11+A12)(00+11) * (B12+B22)(00+11)
    # R4: (A11+A12)10 * B12 00
    # R5: (A11+A12)10 * B22 00
    # R6: (A11+A12)00 * (B12 01 - B22 11)
    # R7: (A11+A12)11 * (B22 10 - B12 00)
    # R8: (A11+A12)(00+01) * B12 11
    # R9: (A11+A12)(00+01) * B22 11
    # R10: (A11-A12)00 * (B12+B22)(00+01)
    
    # Global A indices: A[i*4+j], A11=[0,1,4,5], A12=[2,3,6,7]
    # Global B indices: B[j*4+k], B11=[0,1,4,5], B12=[2,3,6,7], B21=[8,9,12,13], B22=[10,11,14,15]
    # Global W indices: W[k*2+i]
    
    # Left block (cols 0-1 of C), 10 multiplications at offsets 0-9
    left_mults = [
        # (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
        # L1: (A11+A12)00 * (B11+B21)00
        ([0, 2], [1, 1], [0, 8], [1, 1], [0, 2, 4, 6], [1, -1, 1, -1]),
        # L2: (A11+A12)11 * (B11+B21)11
        ([5, 7], [1, 1], [5, 13], [1, 1], [0, 2, 4, 6], [-1, 1, 1, -1]),
        # L3: (A11+A12)(00+11) * (B11+B21)(00+11)
        ([0, 2, 5, 7], [1, 1, 1, 1], [0, 8, 5, 13], [1, 1, 1, 1], [0, 2, 4, 6], [1, 1, 1, 1]),
        # L4: (A11+A12)10 * B11 00
        ([4, 6], [1, 1], [0], [1], [2, 4], [1, 1]),
        # L5: (A11+A12)10 * B21 00
        ([4, 6], [1, 1], [8], [1], [2, 4], [1, 1]),
        # L6: (A11+A12)00 * (B11 01 - B21 11)
        ([0, 2], [1, 1], [1, 13], [1, -1], [1, 3], [1, 1]),
        # L7: (A11+A12)11 * (B21 10 - B11 00)
        ([5, 7], [1, 1], [12, 0], [1, -1], [0, 2, 4], [1, 1, 1]),
        # L8: (A11+A12)(00+01) * B11 11
        ([0, 1, 2, 3], [1, 1, 1, 1], [5], [1], [0, 1], [-1, 1]),
        # L9: (A11+A12)(00+01) * B21 11
        ([0, 1, 2, 3], [1, 1, 1, 1], [13], [1], [0, 1], [-1, 1]),
        # L10: (A11-A12)00 * (B11+B21)(00+01)
        ([0, 2], [1, -1], [0, 1, 8, 9], [1, 1, 1, 1], [3], [1]),
    ]
    
    # Right block (cols 2-3 of C), 10 multiplications at offsets 10-19
    right_mults = [
        # R1: (A11+A12)00 * (B12+B22)00
        ([0, 2], [1, 1], [2, 10], [1, 1], [4, 6, 7, 9], [1, -1, 1, -1]),
        # R2: (A11+A12)11 * (B12+B22)11
        ([5, 7], [1, 1], [7, 15], [1, 1], [4, 6, 7, 9], [-1, 1, 1, -1]),
        # R3: (A11+A12)(00+11) * (B12+B22)(00+11)
        ([0, 2, 5, 7], [1, 1, 1, 1], [2, 10, 7, 15], [1, 1, 1, 1], [4, 6, 7, 9], [1, 1, 1, 1]),
        # R4: (A11+A12)10 * B12 00
        ([4, 6], [1, 1], [2], [1], [6, 8], [1, 1]),
        # R5: (A11+A12)10 * B22 00
        ([4, 6], [1, 1], [10], [1], [6, 8], [1, 1]),
        # R6: (A11+A12)00 * (B12 01 - B22 11)
        ([0, 2], [1, 1], [3, 15], [1, -1], [5, 7], [1, 1]),
        # R7: (A11+A12)11 * (B22 10 - B12 00)
        ([5, 7], [1, 1], [14, 2], [1, -1], [4, 6, 8], [1, 1, 1]),
        # R8: (A11+A12)(00+01) * B12 11
        ([0, 1, 2, 3], [1, 1, 1, 1], [7], [1], [4, 5], [-1, 1]),
        # R9: (A11+A12)(00+01) * B22 11
        ([0, 1, 2, 3], [1, 1, 1, 1], [15], [1], [4, 5], [-1, 1]),
        # R10: (A11-A12)00 * (B12+B22)(00+01)
        ([0, 2], [1, -1], [2, 3, 10, 11], [1, 1, 1, 1], [7], [1]),
    ]
    
    # Fill in U, V, W
    for offset, mults in [(0, left_mults), (10, right_mults)]:
        for s, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(mults):
            col = offset + s
            for idx, c in zip(u_idx, u_c):
                U[idx][col] += c
            for idx, c in zip(v_idx, v_c):
                V[idx][col] += c
            for idx, c in zip(w_idx, w_c):
                W[idx][col] += c
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to rank 28 if rank 24 fails
        rank28 = 28
        U28 = [[0] * rank28 for _ in range(n * m)]
        V28 = [[0] * rank28 for _ in range(m * p)]
        W28 = [[0] * rank28 for _ in range(p * n)]
        
        strassen_steps = [
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        ]
        
        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s
                for local_idx, coeff in step['u'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_idx = l_row * m + (a_col_start + l_col)
                    U28[global_idx][col] += coeff
                for local_idx, coeff in step['v'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_idx = (b_row_start + l_row) * p + (b_col_start + l_col)
                    V28[global_idx][col] += coeff
                for local_idx, coeff in step['w'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_idx = (c_col_start + l_col) * n + l_row
                    W28[global_idx][col] += coeff
        
        add_block(0, 0, 0, 0, 0)
        add_block(0, 0, 2, 2, 7)
        add_block(2, 2, 0, 0, 14)
        add_block(2, 2, 2, 2, 21)
        
        return {"U": U28, "V": V28, "W": W28}
    
    return {"U": U, "V": V, "W": W}
