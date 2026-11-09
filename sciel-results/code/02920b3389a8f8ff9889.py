import time
import numpy as np

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140

    # 1. Define Strassen 2x2 vectors
    # U, V, W are length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]

    # 2. Construct 49 terms for M_4,4,4
    # U_4x4, V_4x4, W_4x4 will be lists of length 49, each a list of length 16
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for m in range(7):
        for s in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # Expand U
            # U_strassen[m] is block coefficients (length 4)
            # U_strassen[s] is scalar coefficients (length 4)
            for a in range(2): # block row
                for x in range(2): # scalar row
                    # Block index a, scalar index x
                    # Global row i = 2a + x
                    # U index in 4x4 is 4*i + j? No.
                    # U index for M_4,4,4 is i*m + j = 4i + j.
                    # Here we are building U vector for the tensor.
                    # U[i, j] ? No, U is (nm, rank).
                    # For M_4,4,4, U is (16, rank). Index 4i+j.
                    # But we are constructing the vector for one term.
                    # The term is a rank-1 tensor u ⊗ v ⊗ w.
                    # u is a vector of length 16 (flattened 4x4 matrix A).
                    # u[4i+j] = U_block[a] * U_scalar[x] ?
                    # Wait. A is 4x4.
                    # A_block[a] is a 2x2 matrix.
                    # A_block[a][x, y] = A[2a+x, 2a+y]? No.
                    # A_block[a] corresponds to block row a.
                    # The columns of A_block[a] are 2b.
                    # So A[2a+x, 2b+y].
                    # U vector index is 4*i + j.
                    # i = 2a+x, j = 2b+y.
                    # So u[4*(2a+x) + (2b+y)] = U_block[m][a] * U_scalar[s][x] ?
                    # No. U_block[m][a] is the coefficient for the entire block row a.
                    # U_scalar[s][x] is the coefficient for row x in the 2x2 multiplication.
                    # But the 2x2 multiplication uses the whole 2x2 block.
                    # So for a fixed block row a, the contribution to row 2a+x is U_block[m][a] * U_scalar[s][x].
                    # This applies to all columns in that block row?
                    # Yes, because the block is 2x2, and the scalar multiplication is on the 2x2 blocks.
                    # So for each block column b, and scalar column y:
                    # j = 2b + y.
                    # u[4*(2a+x) + (2b+y)] = U_block[m][a] * U_scalar[s][x]
                    # Note: U_scalar[s][x] does not depend on y or b.
                    
                    val_u = U_strassen[m][a] * U_strassen[s][x]
                    for b in range(2):
                        for y in range(2):
                            idx_u = 4 * (2 * a + x) + (2 * b + y)
                            u_vec[idx_u] = val_u
            
            # Expand V
            # V index 4j + k.
            # j = 2b + y, k = 2c + z (for B)
            # V_block[m][b] * V_scalar[s][y]
            for b in range(2):
                for y in range(2):
                    val_v = V_strassen[m][b] * V_strassen[s][y]
                    for c in range(2):
                        for z in range(2):
                            idx_v = 4 * (2 * b + y) + (2 * c + z)
                            v_vec[idx_v] = val_v
            
            # Expand W
            # W index 4k + i.
            # k = 2c + z, i = 2a + x.
            # W_block[m][c, a]?
            # W_strassen[m] is a 2x2 matrix of coefficients for C_blocks.
            # W_strassen[m][c] is the row of W_block?
            # My W_strassen list is flattened 2x2.
            # Index 0: C00, 1: C01, 2: C10, 3: C11.
            # So W_block[m][c, a] = W_strassen[m][2*c + a]?
            # Let's check M1: W=[1,0,0,1].
            # C00 (a=0,c=0) -> 1. C11 (a=1,c=1) -> 1.
            # Index 2*0+0=0 -> 1. Index 2*1+1=3 -> 1. Correct.
            
            for a in range(2):
                for c in range(2):
                    w_block_val = W_strassen[m][2 * c + a]
                    if w_block_val != 0:
                        for x in range(2):
                            for z in range(2):
                                w_scalar_val = W_strassen[s][2 * z + x]
                                if w_scalar_val != 0:
                                    total_val = w_block_val * w_scalar_val
                                    idx_w = 4 * (2 * c + z) + (2 * a + x)
                                    w_vec[idx_w] = total_val
            
            U_4x4.append(u_vec)
            V_4x4.append(v_vec)
            W_4x4.append(w_vec)

    # 3. Construct M_4,4,8 decomposition (Rank 98)
    rank_total = 98
    U_final = [[0] * rank_total for _ in range(n * m)]
    V_final = [[0] * rank_total for _ in range(m * p)]
    W_final = [[0] * rank_total for _ in range(p * n)]
    
    # First 49 terms: k in 0..3
    for r in range(49):
        for i in range(n * m):
            U_final[i][r] = U_4x4[r][i]
        for j in range(m):
            for k in range(4):
                v_idx = j * p + k
                V_final[v_idx][r] = V_4x4[r][j * 4 + k]
        for k in range(4):
            for i in range(n):
                w_idx = k * n + i
                W_final[w_idx][r] = W_4x4[r][k * 4 + i]
        
        # k in 4..7 are 0 for V and W in first half
        for j in range(m):
            for k in range(4, 8):
                v_idx = j * p + k
                V_final[v_idx][r] = 0
        for k in range(4, 8):
            for i in range(n):
                w_idx = k * n + i
                W_final[w_idx][r] = 0

    # Next 49 terms: k in 4..7
    for r in range(49):
        r_offset = 49 + r
        for i in range(n * m):
            U_final[i][r_offset] = U_4x4[r][i]
        for j in range(m):
            for k in range(4):
                v_idx = j * p + k
                V_final[v_idx][r_offset] = 0
            for k in range(4, 8):
                v_idx = j * p + k
                V_final[v_idx][r_offset] = V_4x4[r][j * 4 + (k - 4)]
        
        for k in range(4):
            for i in range(n):
                w_idx = k * n + i
                W_final[w_idx][r_offset] = 0
        for k in range(4, 8):
            for i in range(n):
                w_idx = k * n + i
                W_final[w_idx][r_offset] = W_4x4[r][(k - 4) * 4 + i]

    return {"U": U_final, "V": V_final, "W": W_final}
