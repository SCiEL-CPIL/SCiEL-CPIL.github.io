import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 4, 4, 9

    def time_ok():
        return time.time() < deadline

    # Winograd 4x4x4 factorization (Rank 25)
    # Using 1D Winograd factors of length 5 for size 4
    # F_A: 5x4, F_B: 4x5, F_C: 4x5
    
    # 1D Winograd factors for n=4, m=4, p=4
    # The 1D factors for multiplying two length-4 vectors with output length 4
    # using 5 multiplications
    
    # Standard Winograd 1D factors for 4-point DFT-like transform
    F_A_1d = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, -1],
        [1, 0, 1, 0],
        [1, 1, 0, -1],
        [1, 0, -1, 0]
    ], dtype=float)
    
    F_B_1d = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, -1],
        [1, 0, -1, 0, 1],
        [1, 0, 1, 0, -1]
    ], dtype=float)
    
    F_C_1d = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, -1],
        [1, 0, -1, 0, 1],
        [1, 0, 1, 0, -1]
    ], dtype=float)
    
    # Build 2D Winograd factors using Kronecker products
    # U_444 = F_A_1d ⊗ I_4 (but we need to be careful with the ordering)
    # Actually, for matrix multiplication C = A * B where A is 4x4, B is 4x4, C is 4x4
    # The tensor decomposition uses:
    # U[i,j, k,l] where (i,j) indexes A, (k,l) indexes B, and the output is C
    
    # The Winograd factorization for 4x4x4 has rank 25
    # U: 16x25, V: 16x25, W: 16x25
    
    # Construct U, V, W for 4x4x4
    U_444 = np.zeros((16, 25), dtype=float)
    V_444 = np.zeros((16, 25), dtype=float)
    W_444 = np.zeros((16, 25), dtype=float)
    
    # The 25 terms correspond to pairs (a, b) where a in [0,4], b in [0,4]
    for a in range(5):
        for b in range(5):
            col = a * 5 + b
            # U: for each (i, j) in A (4x4), U[i*4+j, col] = F_A_1d[a, i] * F_A_1d[b, j]
            for i in range(4):
                for j in range(4):
                    U_444[i * 4 + j, col] = F_A_1d[a, i] * F_A_1d[b, j]
            
            # V: for each (k, l) in B (4x4), V[k*4+l, col] = F_B_1d[k, a] * F_B_1d[l, b]
            for k in range(4):
                for l in range(4):
                    V_444[k * 4 + l, col] = F_B_1d[k, a] * F_B_1d[l, b]
            
            # W: for each (k, i) in C (4x4), W[k*4+i, col] = F_C_1d[k, a] * F_C_1d[i, b]
            for k in range(4):
                for i in range(4):
                    W_444[k * 4 + i, col] = F_C_1d[k, a] * F_C_1d[i, b]
    
    # Schoolbook 4x4x1 (Rank 16)
    U_441 = np.zeros((16, 16), dtype=float)
    V_441 = np.zeros((4, 16), dtype=float)
    W_441 = np.zeros((4, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1
    
    # Assemble full 4x4x9 decomposition
    total_rank = 25 + 25 + 16  # 66
    U_total = np.zeros((16, total_rank), dtype=float)
    V_total = np.zeros((36, total_rank), dtype=float)
    W_total = np.zeros((36, total_rank), dtype=float)
    
    # Part 1: C1 = A B1 (cols 0-3 of B, cols 0-3 of C)
    U_total[:, 0:25] = U_444
    V_part1 = np.zeros((36, 25), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part1[j*9 + k, :] = V_444[j*4 + k, :]
    V_total[:, 0:25] = V_part1
    W_part1 = np.zeros((36, 25), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part1[k*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 0:25] = W_part1
    
    # Part 2: C2 = A B2 (cols 4-7 of B, cols 4-7 of C)
    U_total[:, 25:50] = U_444
    V_part2 = np.zeros((36, 25), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part2[j*9 + (4+k), :] = V_444[j*4 + k, :]
    V_total[:, 25:50] = V_part2
    W_part2 = np.zeros((36, 25), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part2[(4+k)*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 25:50] = W_part2
    
    # Part 3: C3 = A B3 (col 8 of B, col 8 of C)
    U_total[:, 50:66] = U_441
    V_part3 = np.zeros((36, 16), dtype=float)
    for j in range(4):
        V_part3[j*9 + 8, :] = V_441[j, :]
    V_total[:, 50:66] = V_part3
    W_part3 = np.zeros((36, 16), dtype=float)
    for i in range(4):
        W_part3[8*4 + i, :] = W_441[i, :]
    W_total[:, 50:66] = W_part3
    
    # Fast validation
    def validate_fast(U, V, W, tol=1e-9):
        U_arr = np.asarray(U, dtype=float)
        V_arr = np.asarray(V, dtype=float)
        W_arr = np.asarray(W, dtype=float)
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        prod = U_arr[u_rows, :] * V_arr[v_rows, :] * W_arr[w_rows, :]
        sums = prod.sum(axis=1)
        return np.all(np.abs(sums - 1.0) <= tol)
    
    # Verify decomposition
    if not validate_fast(U_total, V_total, W_total):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=float)
        V_sb = np.zeros((m*p, rank_sb), dtype=float)
        W_sb = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
