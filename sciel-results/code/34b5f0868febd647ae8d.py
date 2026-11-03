import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return DEADLINE - elapsed()
    
    # Build Strassen 2x2x2 decomposition
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11) * B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00 * (B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11 * (B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01) * B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen terms for 2x2x2: (A_blocks, B_blocks, C_blocks)
    # Each block is (row_block, col_block, coefficient)
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00 + C11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10 + A11) * B00 -> C10
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A00 * (B01 - B11) -> C01
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        # M4 = A11 * (B10 - B00) -> C00 + C10
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A00 + A01) * B11 -> -C00 + C01
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01) -> C11
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11) -> C00 + C11
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    # For each Strassen term at the 4x4 level, we apply the 2x2 Strassen
    # at each of the 7 sub-blocks (2x2 blocks within the 4x4)
    for l in range(7):
        A_blocks, B_blocks, C_blocks = strassen_terms[l]
        for s in range(7):
            col = l * 7 + s
            # For U: A is 4x4, blocks are 2x2
            for (br, bc, coeff) in A_blocks:
                for ii in range(4):
                    val = U2[ii, s]
                    if val != 0:
                        # ii maps to (block_row, block_col) within 2x2
                        br_local = ii // 2
                        bc_local = ii % 2
                        r = br * 2 + br_local
                        c = bc * 2 + bc_local
                        U4[r * 4 + c, col] += coeff * val
            # For V: B is 4x4, blocks are 2x2
            for (br, bc, coeff) in B_blocks:
                for ii in range(4):
                    val = V2[ii, s]
                    if val != 0:
                        br_local = ii // 2
                        bc_local = ii % 2
                        r = br * 2 + br_local
                        c = bc * 2 + bc_local
                        V4[r * 4 + c, col] += coeff * val
            # For W: C is 4x4, blocks are 2x2
            for (br, bc, coeff) in C_blocks:
                for ii in range(4):
                    val = W2[ii, s]
                    if val != 0:
                        br_local = ii // 2
                        bc_local = ii % 2
                        r = br * 2 + br_local
                        c = bc * 2 + bc_local
                        W4[r * 4 + c, col] += coeff * val
    
    # Verify 4x4x4 decomposition
    def verify_core():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    core_ok = verify_core()
    
    if not core_ok:
        # Fallback: schoolbook for 4x4x4
        U4 = np.zeros((16, 64))
        V4 = np.zeros((16, 64))
        W4 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
        core_rank = 64
    else:
        core_rank = 49
    
    # Build full 4x5 x 5x5 decomposition
    # Part 1: core (4x4x4) - rank core_rank
    # Part 2: j in 0..3, k=4 (4x4x1) - rank 16
    # Part 3: j=4, k in 0..4 (4x1x5) - rank 20
    total_rank = core_rank + 16 + 20
    
    U_full = np.zeros((20, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((20, total_rank))
    
    # Copy core into full matrices
    for i in range(4):
        for j in range(4):
            U_full[i * 5 + j, :core_rank] = U4[i * 4 + j, :]
    
    for j in range(4):
        for k in range(4):
            V_full[j * 5 + k, :core_rank] = V4[j * 4 + k, :]
    
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :core_rank] = W4[k * 4 + i, :]
    
    # Part 2: sum_{j=0}^{3} A[i,j] * B[j,4] for C[i,4]
    for i in range(4):
        for j in range(4):
            r = core_rank + i * 4 + j
            U_full[i * 5 + j, r] = 1
            V_full[j * 5 + 4, r] = 1
            W_full[4 * 4 + i, r] = 1
    
    # Part 3: A[i,4] * B[4,k] for C[i,k], k in 0..4
    for i in range(4):
        for k in range(5):
            r = core_rank + 16 + i * 5 + k
            U_full[i * 5 + 4, r] = 1
            V_full[4 * 5 + k, r] = 1
            W_full[k * 4 + i, r] = 1
    
    # Verify full decomposition
    def verify_full():
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    val = np.dot(U_full[i * 5 + j, :], V_full[j * 5 + k, :] * W_full[k * 4 + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify_full():
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook rank 100
        rank = 100
        U = np.zeros((20, rank))
        V = np.zeros((25, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j, col] = 1
                    V[j * 5 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
