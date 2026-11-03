import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (25, rank)  
    # W shape: (p*n, rank) = (20, rank)
    # W index: k*n + i (trace convention)
    
    # Strategy: Split the multiplication into parts
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Part 1: j in 0..3, k in 0..3 -> 4x4x4 core, use Strassen (rank 49)
    # Part 2: j in 0..3, k=4 -> 4x4x1, schoolbook (rank 16)
    # Part 3: j=4, k in 0..4 -> 4x1x5, schoolbook (rank 20)
    # Total: 49 + 16 + 20 = 85
    
    # Build Strassen 2x2x2 decomposition
    def strassen_2x2():
        # U, V, W each shape (4, 7)
        # Index: row*2 + col for 2x2 matrices
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A indices: 00->0, 01->1, 10->2, 11->3
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
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    # Index: row*4 + col for 4x4 matrices
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen 4x4: 7 terms, each is a 2x2x2 product
    # For term l, the A and B blocks are specified as (block_row, block_col, coeff)
    # C blocks specify where the result goes
    
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10 + A11) * B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A00 * (B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        # M4 = A11 * (B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A00 + A01) * B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    for l, (A_blocks, B_blocks, C_blocks) in enumerate(strassen_terms):
        for s in range(7):
            col = l * 7 + s
            
            # Add contributions to U4
            for (br, bc, coeff) in A_blocks:
                for ii in range(4):
                    val = U2[ii, s]
                    if val != 0:
                        r = br * 2 + ii // 2
                        c = bc * 2 + ii % 2
                        U4[r * 4 + c, col] += coeff * val
            
            # Add contributions to V4
            for (br, bc, coeff) in B_blocks:
                for ii in range(4):
                    val = V2[ii, s]
                    if val != 0:
                        r = br * 2 + ii // 2
                        c = bc * 2 + ii % 2
                        V4[r * 4 + c, col] += coeff * val
            
            # Add contributions to W4
            for (br, bc, coeff) in C_blocks:
                for ii in range(4):
                    val = W2[ii, s]
                    if val != 0:
                        r = br * 2 + ii // 2
                        c = bc * 2 + ii % 2
                        W4[r * 4 + c, col] += coeff * val
    
    # Verify 4x4x4 decomposition
    def verify_4x4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    core_ok = verify_4x4x4()
    
    if not core_ok:
        # Fallback: use schoolbook for 4x4x4 (rank 64)
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
    # U_full[i*5+j, r] = U4[i*4+j, r] for i,j in 0..3
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
    # 16 terms: one for each (i,j) pair
    for i in range(4):
        for j in range(4):
            r = core_rank + i * 4 + j
            U_full[i * 5 + j, r] = 1
            V_full[j * 5 + 4, r] = 1
            W_full[4 * 4 + i, r] = 1  # k=4, W index = 4*4 + i
    
    # Part 3: A[i,4] * B[4,k] for C[i,k], k in 0..4
    # 20 terms: one for each (i,k) pair
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
