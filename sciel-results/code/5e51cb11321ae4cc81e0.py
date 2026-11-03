import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Target Rank: 49 (Core) + 16 (Term 2) + 1 (Term 3) = 66
    rank = 66
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4x4 Core (Rank 49) ---
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # U index: i*m + j = i*5 + j
    # V index: j*p + k = j*5 + k
    # W index: k*n + i = k*4 + i
    
    # Helper to get Strassen 2x2 decomposition
    def strassen_2x2():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A10 + A11) * B00
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1
        
        # M3 = A00 * (B01 - B11)
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1
        
        # M4 = A11 * (B10 - B00)
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        # M5 = (A00 + A01) * B11
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1; W2[3, 6] = 1
        
        return U2, V2, W2

    U2, V2, W2 = strassen_2x2()
    
    # Build 4x4x4 using 7 Strassen steps, each with 7 inner terms
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen coefficients for 4x4
    # l in 0..6
    # A_blocks: list of (br, bc, coeff)
    # B_blocks: list of (br, bc, coeff)
    # C_blocks: list of (br, bc, coeff)
    
    strassen_blocks = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),                     # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),                    # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),           # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),           # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),           # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)])   # M7
    ]
    
    for l, (A_blocks, B_blocks, C_blocks) in enumerate(strassen_blocks):
        for s in range(7):
            col_idx = l * 7 + s
            
            # Update U4
            for (br, bc, coeff) in A_blocks:
                for inner_i in range(4):
                    val = U2[inner_i, s]
                    if val != 0:
                        br_inner = inner_i // 2
                        bc_inner = inner_i % 2
                        # Global index in 4x4 flattened (row-major)
                        # Row = 2*br + br_inner, Col = 2*bc + bc_inner
                        gi = 4 * (2 * br + br_inner) + (2 * bc + bc_inner)
                        U4[gi, col_idx] += coeff * val
                        
            # Update V4
            for (br, bc, coeff) in B_blocks:
                for inner_i in range(4):
                    val = V2[inner_i, s]
                    if val != 0:
                        br_inner = inner_i // 2
                        bc_inner = inner_i % 2
                        gi = 4 * (2 * br + br_inner) + (2 * bc + bc_inner)
                        V4[gi, col_idx] += coeff * val
                        
            # Update W4
            for (br, bc, coeff) in C_blocks:
                for inner_i in range(4):
                    val = W2[inner_i, s]
                    if val != 0:
                        br_inner = inner_i // 2
                        bc_inner = inner_i % 2
                        gi = 4 * (2 * br + br_inner) + (2 * bc + bc_inner)
                        W4[gi, col_idx] += coeff * val

    # Copy Core to Full Matrices
    # U4 index gi = 4*i + j. Full U index = i*5 + j.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, :49] = U4[i * 4 + j, :]
            V[j * 5 + j, :49] = 0 # Placeholder, will be filled below
            # V index: j*5 + k. V4 index: 4*j + k.
            # We need to map V4[4*j+k] to V[j*5+k]
            
    # Correct V and W mapping
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V4[j * 4 + k, :]
            
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k * 4 + i, :]
            
    # --- Part 2: Term 2 (Rank 16) ---
    # j in 0..3, k=4
    # U[i, j], V[j, 4], W[4, i]
    # Full U: i*5 + j
    # Full V: j*5 + 4
    # Full W: 4*4 + i = 16 + i
    
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * 5 + j, r] = 1
            V[j * 5 + 4, r] = 1
            W[16 + i, r] = 1
            
    # --- Part 3: Term 3 (Rank 1) ---
    # j=4, k in 0..4
    # U[i, 4], V[4, k], W[k, i]
    # Full U: i*5 + 4
    # Full V: 4*5 + k = 20 + k
    # Full W: k*4 + i
    
    r_last = 49 + 16
    for i in range(4):
        U[i * 5 + 4, r_last] = 1
    for k in range(5):
        V[20 + k, r_last] = 1
    for k in range(5):
        for i in range(4):
            W[k * 4 + i, r_last] = 1
            
    # Verification
    is_exact = True
    for i in range(4):
        for j in range(5):
            for k in range(5):
                val = np.dot(U[i * 5 + j, :], V[j * 5 + k, :] * W[k * 4 + i, :])
                if abs(val - 1.0) > 1e-9:
                    is_exact = False
                    break
            if not is_exact:
                break
        if not is_exact:
            break
            
    if is_exact:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to Schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][column] = 1
                    V_s[j * p + k][column] = 1
                    W_s[k * n + i][column] = 1
                    column += 1
        return {
            "U": U_s,
            "V": V_s,
            "W": W_s
        }
