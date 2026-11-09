import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively.
    # Rank is 88.
    # U: (n*m, 88), V: (m*p, 88), W: (p*n, 88)
    
    U = np.zeros((n * m, 88), dtype=int)
    V = np.zeros((m * p, 88), dtype=int)
    W = np.zeros((p * n, 88), dtype=int)
    
    rank_offset = 0
    
    # Helper to add a sub-decomposition
    # sub_n, sub_m, sub_p: dimensions of the sub-problem
    # i_off, j_off, k_off: offsets for indices in global A, B, C
    # U_sub, V_sub, W_sub: factors for the sub-problem
    def add_sub_problem(sub_n, sub_m, sub_p, i_off, j_off, k_off, U_sub, V_sub, W_sub):
        nonlocal rank_offset
        r = U_sub.shape[1]
        
        # Map U indices: global a = i*m + j
        # sub a = si * sub_m + sj
        # global i = i_off + si, global j = j_off + sj
        for si in range(sub_n):
            for sj in range(sub_m):
                global_i = i_off + si
                global_j = j_off + sj
                global_a = global_i * m + global_j
                sub_a = si * sub_m + sj
                U[global_a, rank_offset:rank_offset+r] = U_sub[sub_a, :]
                
        # Map V indices: global b = j*p + k
        # sub b = sj * sub_p + sk
        # global j = j_off + sj, global k = k_off + sk
        for sj in range(sub_m):
            for sk in range(sub_p):
                global_j = j_off + sj
                global_k = k_off + sk
                global_b = global_j * p + global_k
                sub_b = sj * sub_p + sk
                V[global_b, rank_offset:rank_offset+r] = V_sub[sub_b, :]
                
        # Map W indices: global c = k*n + i
        # sub c = sk * sub_n + si
        # global k = k_off + sk, global i = i_off + si
        for sk in range(sub_p):
            for si in range(sub_n):
                global_k = k_off + sk
                global_i = i_off + si
                global_c = global_k * n + global_i
                sub_c = sk * sub_n + si
                W[global_c, rank_offset:rank_offset+r] = W_sub[sub_c, :]
                
        rank_offset += r

    # 1. Split m=4 into 2+2
    # Block 1: A[:, 0:2] * B[0:2, :] -> 3x2x8
    # Block 2: A[:, 2:4] * B[2:4, :] -> 3x2x8
    
    # For each 3x2x8 block, split p=8 into 4+4
    # For each 3x2x4 block, split n=3 into 1+2
    # 1x2x4: Schoolbook (Rank 8)
    # 2x2x4: Split p=4 into 2+2 -> 2x2x2 (Strassen Rank 7) + 2x2x2 (Strassen Rank 7)
    
    # Define Strassen 2x2x2
    U_str = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0]
    ], dtype=int)
    # Wait, Strassen U, V, W need to be consistent.
    # Let's use the standard Strassen coefficients.
    # M1 = A00(B01+B11) -> U=[1,0,0,0], V=[0,1,1,0], W=[1,0,0,1] (for C00, C01, C10, C11 parts)
    # This is getting complex to hardcode. I will generate them programmatically or use a known set.
    
    # Standard Strassen:
    # M1 = A00 * (B01 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * (B00 + B01)
    # M6 = (A11 - A00) * (B10 + B11)
    # M7 = (A01 - A11) * (B00 + B10)
    
    # C00 = M1 - M5 + M6 + M7
    # C01 = M3 - M4 + M5 - M7
    # C10 = M2 - M3 + M4 + M6
    # C11 = M1 + M2 - M4 + M5
    
    # U (4x7):
    # Row 0 (A00): 1, 0, 1, 0, 1, 1, 0
    # Row 1 (A01): 0, 0, 1, 0, 0, 0, -1
    # Row 2 (A10): 0, 1, 0, 0, 1, 0, 0
    # Row 3 (A11): 0, 1, 0, 1, 0, -1, -1
    U_strassen = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, -1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, -1, -1]
    ], dtype=int)
    
    # V (4x7):
    # Col 0 (M1): 0, 1, 1, 0 (B01, B11) -> B indices 1, 3? 
    # B is 2x2. Indices: 00, 01, 10, 11.
    # M1 uses B01, B11. V col 0: [0, 1, 0, 1]
    # M2 uses B00. V col 1: [1, 0, 0, 0]
    # M3 uses B11. V col 2: [0, 0, 0, 1]
    # M4 uses B10, B00. V col 3: [-1, 0, 1, 0]
    # M5 uses B00, B01. V col 4: [1, 1, 0, 0]
    # M6 uses B10, B11. V col 5: [0, 0, 1, 1]
    # M7 uses B00, B10. V col 6: [1, 0, 1, 0]
    
    V_strassen = np.array([
        [0, 1, 0, -1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # W (4x7):
    # C00 (idx 0): M1 - M5 + M6 + M7 -> [1, 0, 0, 0, -1, 1, 1]
    # C01 (idx 1): M3 - M4 + M5 - M7 -> [0, 0, 1, -1, 1, 0, -1]
    # C10 (idx 2): M2 - M3 + M4 + M6 -> [0, 1, -1, 1, 0, 1, 0]
    # C11 (idx 3): M1 + M2 - M4 + M5 -> [1, 1, 0, -1, 1, 0, 0]
    
    W_strassen = np.array([
        [1, 0, 0, 0, -1, 1, 1],
        [0, 0, 1, -1, 1, 0, -1],
        [0, 1, -1, 1, 0, 1, 0],
        [1, 1, 0, -1, 1, 0, 0]
    ], dtype=int)

    # Helper for Schoolbook 1x2x4
    # A (1x2), B (2x4), C (1x4)
    # Rank 8.
    # U (2x8): Identity-like?
    # U[0, 0:2] = 1? No.
    # U has shape (1*2, 8) = (2, 8).
    # V has shape (2*4, 8) = (8, 8).
    # W has shape (4*1, 8) = (4, 8).
    
    # Term r corresponds to A[j] * B[j, k].
    # j in 0,1. k in 0..3.
    # U[j, r] = 1 if j == r//4 ?
    # Let's iterate r from 0 to 7.
    # j = r // 4. k = r % 4.
    # U[j, r] = 1.
    # V[j*4+k, r] = 1.
    # W[k*1+0, r] = 1.
    
    U_124 = np.zeros((2, 8), dtype=int)
    V_124 = np.zeros((8, 8), dtype=int)
    W_124 = np.zeros((4, 8), dtype=int)
    for r in range(8):
        j = r // 4
        k = r % 4
        U_124[j, r] = 1
        V_124[j*4+k, r] = 1
        W_124[k, r] = 1

    # Now build the 3x2x4 block
    # n=3, m=2, p=4.
    # Split n into 1+2.
    # Block A: 1x2x4 (Rank 8)
    # Block B: 2x2x4 (Rank 14)
    
    # U_324 (6x22), V_324 (8x22), W_324 (12x22)
    U_324 = np.zeros((6, 22), dtype=int)
    V_324 = np.zeros((8, 22), dtype=int)
    W_324 = np.zeros((12, 22), dtype=int)
    
    # Add 1x2x4 at i_off=0
    # U_124 maps to U_324[0:2, 0:8]
    U_324[0:2, 0:8] = U_124
    # V_124 maps to V_324[0:8, 0:8]
    V_324[0:8, 0:8] = V_124
    # W_124 maps to W_324[0:4, 0:8]
    W_324[0:4, 0:8] = W_124
    
    # Add 2x2x4 at i_off=1
    # 2x2x4 is two 2x2x2.
    # U_224 (4x14), V_224 (8x14), W_224 (8x14)
    U_224 = np.zeros((4, 14), dtype=int)
    V_224 = np.zeros((8, 14), dtype=int)
    W_224 = np.zeros((8, 14), dtype=int)
    
    # First 2x2x2 (k_off=0)
    U_224[0:4, 0:7] = U_strassen
    V_224[0:4, 0:7] = V_strassen
    W_224[0:4, 0:7] = W_strassen
    
    # Second 2x2x2 (k_off=2)
    U_224[0:4, 7:14] = U_strassen
    V_224[4:8, 7:14] = V_strassen
    W_224[4:8, 7:14] = W_strassen
    
    # Embed 2x2x4 into 3x2x4
    # U_224 maps to U_324[2:6, 8:22]
    U_324[2:6, 8:22] = U_224
    # V_224 maps to V_324[0:8, 8:22]
    V_324[0:8, 8:22] = V_224
    # W_224 maps to W_324[4:12, 8:22]
    W_324[4:12, 8:22] = W_224

    # Now build the full 3x4x8
    # We have 4 blocks of 3x2x4.
    # Block 1: i_off=0, j_off=0, k_off=0
    # Block 2: i_off=0, j_off=0, k_off=4
    # Block 3: i_off=0, j_off=2, k_off=0
    # Block 4: i_off=0, j_off=2, k_off=4
    
    # Wait, the split was:
    # m=4 into 2+2.
    # p=8 into 4+4.
    # So we have 4 sub-problems of 3x2x4.
    
    add_sub_problem(3, 2, 4, 0, 0, 0, U_324, V_324, W_324)
    add_sub_problem(3, 2, 4, 0, 0, 4, U_324, V_324, W_324)
    add_sub_problem(3, 2, 4, 0, 2, 0, U_324, V_324, W_324)
    add_sub_problem(3, 2, 4, 0, 2, 4, U_324, V_324, W_324)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
