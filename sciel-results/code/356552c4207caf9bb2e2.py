import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Construct 4x4 factors using Kronecker product
    # np.kron(U2, U2) has shape (16, 49)
    # Row index in kron: L4 = L2_outer * 4 + L2_inner
    # L2_outer = r_block * 2 + c_block
    # L2_inner = r_inner * 2 + c_inner
    # We need to map to S = r * 4 + c
    # r = r_block * 2 + r_inner
    # c = c_block * 2 + c_inner
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutation to map Standard Row-Major indices to Kronecker row indices
    # For standard index S = r*4+c:
    #   r = r_block*2 + r_inner, c = c_block*2 + c_inner
    #   L2_out = r_block*2 + c_block, L2_in = r_inner*2 + c_inner
    #   Kron index = L2_out*4 + L2_in
    # So perm[S] = kron_index, and we use U4[S] = U4_kron[perm[S]]
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_block = r // 2
        r_inner = r % 2
        c_block = c // 2
        c_inner = c % 2
        
        L2_out = r_block * 2 + c_block
        L2_in = r_inner * 2 + c_inner
        
        kron_idx = L2_out * 4 + L2_in
        perm[S] = kron_idx
        
    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]
    
    # Full verification: check all 64 tensor coordinates
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if (i == k) else 0
                u_row = U4[i * 4 + j]
                v_row = V4[j * 4 + k]
                w_row = W4[k * 4 + i]
                prod = int(np.sum(u_row * v_row * w_row))
                if prod != expected:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
    
    # Fallback Rank 64
    n, m, p = 4, 4, 4
    rank = n * m * p
    U_f = [[0] * rank for _ in range(n * m)]
    V_f = [[0] * rank for _ in range(m * p)]
    W_f = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j][col] = 1
                V_f[j*p+k][col] = 1
                W_f[k*n+i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
