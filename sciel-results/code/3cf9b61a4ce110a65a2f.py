# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 9
    
    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    # U, V, W are 4x7
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # m1: (a+d)(e+h) -> C00+C11
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # m2: (c+d)e -> C10+C11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1
    
    # m3: a(f-h) -> C01+C11
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # m4: d(g-e) -> C00+C11
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[3, 3] = 1
    
    # m5: (a+b)h -> C00-C01
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = -1
    
    # m6: (c-a)(e+f) -> C10
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[2, 5] = 1
    
    # m7: (b-d)(g+h) -> C00+C01
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[1, 6] = 1
    
    # 2. Construct 4x4x4 Strassen decomposition (Rank 49)
    # 7 blocks M1..M7, each 2x2x2 (Rank 7)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Mapping of 4x4 indices to 2x2 block indices
    # A block (r,c) has rows 2r..2r+1, cols 2c..2c+1
    # Index in 4x4: i*4 + j
    # Index in 2x2: i'*2 + j'
    
    def map_indices_4to2(i, j):
        # Returns list of 4 indices in 4x4 matrix corresponding to the 2x2 block at (i//2, j//2)
        # Actually, we need to map the local 2x2 indices to global 4x4 indices
        pass

    # We define the Strassen formulas for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # For each M_l, we add the 2x2x2 Strassen terms scaled and shifted.
    
    # Helper to add a 2x2x2 term to the 4x4x4 matrices
    def add_block_term(U_target, V_target, W_target, 
                       U_block, V_block, W_block,
                       a_blocks, b_blocks, c_blocks,
                       scale, col_offset):
        # a_blocks: list of (r, c) blocks for A, with signs
        # b_blocks: list of (r, c) blocks for B, with signs
        # c_blocks: list of (r, c) blocks for C, with signs
        
        for s in range(7):
            col = col_offset + s
            
            # U
            for (br, bc, sign) in a_blocks:
                # Block (br, bc) in 2x2 grid of 2x2 blocks
                # Rows: 2*br, 2*br+1
                # Cols: 2*bc, 2*bc+1
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        u_idx = i_global * 4 + j_global
                        u_local_idx = i_local * 2 + j_local
                        U_target[u_idx, col] += sign * U_block[u_local_idx, s]
                        
            # V
            for (br, bc, sign) in b_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        v_idx = i_global * 4 + j_global
                        v_local_idx = i_local * 2 + j_local
                        V_target[v_idx, col] += sign * V_block[v_local_idx, s]
                        
            # W
            for (br, bc, sign) in c_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        w_idx = i_global * 4 + j_global
                        w_local_idx = i_local * 2 + j_local
                        W_target[w_idx, col] += sign * W_block[w_local_idx, s]

    # M1: (A00 + A11)(B00 + B11) -> C00 + C11
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,0,1), (1,1,1)], 
                   [(0,0,1), (1,1,1)], 
                   [(0,0,1), (1,1,1)],
                   1.0, 0)
                   
    # M2: (A10 + A11)B00 -> C10 + C11
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(1,0,1), (1,1,1)], 
                   [(0,0,1)], 
                   [(1,0,1), (1,1,1)],
                   1.0, 7)
                   
    # M3: A00(B01 - B11) -> C01 + C11
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,0,1)], 
                   [(0,1,1), (1,1,-1)], 
                   [(0,1,1), (1,1,1)],
                   1.0, 14)
                   
    # M4: A11(B10 - B00) -> C00 + C11
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(1,1,1)], 
                   [(1,0,1), (0,0,-1)], 
                   [(0,0,1), (1,1,1)],
                   1.0, 21)
                   
    # M5: (A00 + A01)B11 -> C00 - C01
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,0,1), (0,1,1)], 
                   [(1,1,1)], 
                   [(0,0,1), (0,1,-1)],
                   1.0, 28)
                   
    # M6: (A10 - A00)(B00 + B01) -> C10
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(1,0,1), (0,0,-1)], 
                   [(0,0,1), (0,1,1)], 
                   [(1,0,1)],
                   1.0, 35)
                   
    # M7: (A01 - A11)(B10 + B11) -> C00 + C01
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,1,1), (1,1,-1)], 
                   [(1,0,1), (1,1,1)], 
                   [(0,0,1), (0,1,1)],
                   1.0, 42)

    # 3. Construct 4x4x1 Schoolbook decomposition (Rank 16)
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1

    # 4. Assemble full 4x4x9 decomposition
    # Total rank: 49 (for B1) + 49 (for B2) + 16 (for B3) = 114
    
    U_total = np.zeros((16, 114))
    V_total = np.zeros((36, 114)) # m*p = 4*9 = 36
    W_total = np.zeros((36, 114)) # p*n = 9*4 = 36
    
    # Part 1: B1 (cols 0-3)
    # U is same as U_444
    U_total[:, 0:49] = U_444
    
    # V: indices j*9 + k. k in 0..3
    # V_444 has indices j*4 + k (local).
    # We need to map local (j, k) to global (j, k) in 4x9.
    # Local j is 0..3, local k is 0..3.
    # Global row index = j*9 + k.
    V_part1 = np.zeros((36, 49))
    for j in range(4):
        for k in range(4):
            V_part1[j*9 + k, :] = V_444[j*4 + k, :]
    V_total[:, 0:49] = V_part1
    
    # W: indices k*4 + i. k in 0..3.
    # W_444 has indices k*4 + i (local).
    # Global row index = k*4 + i.
    W_part1 = np.zeros((36, 49))
    for k in range(4):
        for i in range(4):
            W_part1[k*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 0:49] = W_part1
    
    # Part 2: B2 (cols 4-7)
    U_total[:, 49:98] = U_444
    
    V_part2 = np.zeros((36, 49))
    for j in range(4):
        for k in range(4):
            # Global k is 4..7
            V_part2[j*9 + (4+k), :] = V_444[j*4 + k, :]
    V_total[:, 49:98] = V_part2
    
    W_part2 = np.zeros((36, 49))
    for k in range(4):
        for i in range(4):
            # Global k is 4..7
            W_part2[(4+k)*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 49:98] = W_part2
    
    # Part 3: B3 (col 8)
    U_total[:, 98:114] = U_441
    
    V_part3 = np.zeros((36, 16))
    for j in range(4):
        V_part3[j*9 + 8, :] = V_441[j, :]
    V_total[:, 98:114] = V_part3
    
    W_part3 = np.zeros((36, 16))
    for i in range(4):
        W_part3[8*4 + i, :] = W_441[i, :]
    W_total[:, 98:114] = W_part3
    
    # Validation
    def validate(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if validate(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
