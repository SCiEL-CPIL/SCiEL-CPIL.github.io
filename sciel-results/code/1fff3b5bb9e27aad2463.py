import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get Strassen decomposition for s x s x s
    def get_strassen_decomp(s):
        if s == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        half = s // 2
        U_s, V_s, W_s = get_strassen_decomp(half)
        rank_s = len(U_s[0])
        rank = 7 * rank_s
        
        U = [[0] * rank for _ in range(s * s)]
        V = [[0] * rank for _ in range(s * s)]
        W = [[0] * rank for _ in range(s * s)]
        
        # Strassen coefficients
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # alpha[i][a][b] is coeff of A_ab in M_i
        # beta[i][c][d] is coeff of B_cd in M_i
        # gamma[i][a][b] is coeff of M_i in C_ab
        
        alpha = [
            [[1, 0], [0, 1]], # M1: A11 + A22
            [[0, 0], [1, 1]], # M2: A21 + A22
            [[1, 0], [0, 0]], # M3: A11
            [[0, 0], [0, 1]], # M4: A22
            [[1, 1], [0, 0]], # M5: A11 + A12
            [[-1, 0], [1, 0]],# M6: A21 - A11
            [[0, 1], [0, -1]] # M7: A12 - A22
        ]
        
        beta = [
            [[1, 0], [0, 1]], # M1: B11 + B22
            [[1, 0], [0, 0]], # M2: B11
            [[0, 1], [0, -1]],# M3: B12 - B22
            [[0, -1], [1, 0]],# M4: B21 - B11
            [[0, 0], [0, 1]], # M5: B22
            [[1, 1], [0, 0]], # M6: B11 + B12
            [[0, 1], [1, 1]]  # M7: B21 + B22
        ]
        
        gamma = [
            [[1, 0], [0, 1]], # C11 += M1, C22 += M1
            [[0, 0], [1, -1]],# C21 += M2, C22 -= M2
            [[0, 1], [0, 0]], # C12 += M3
            [[1, 0], [0, 1]], # C11 += M4, C22 += M4
            [[-1, 0], [0, 0]],# C11 -= M5
            [[0, 0], [0, 1]], # C22 += M6
            [[1, 0], [0, 0]]  # C11 += M7
        ]
        # Correction on gamma:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        gamma = [
            [[1, 0], [0, 1]], # M1: C11, C22
            [[0, 0], [1, -1]],# M2: C21, C22
            [[0, 1], [0, 0]], # M3: C12
            [[1, 0], [0, 1]], # M4: C11, C22
            [[-1, 1], [0, 0]],# M5: C11, C12
            [[0, 0], [0, 1]], # M6: C22
            [[1, 0], [0, 0]]  # M7: C11
        ]
        
        for i in range(7):
            for k in range(rank_s):
                r = i * rank_s + k
                for u in range(half):
                    for v in range(half):
                        local_idx = u * half + v
                        
                        # Update U
                        for a in range(2):
                            for b in range(2):
                                coeff = alpha[i][a][b]
                                if coeff != 0:
                                    global_row = a * half + u
                                    global_col = b * half + v
                                    global_idx = global_row * s + global_col
                                    U[global_idx][r] += coeff * U_s[local_idx][k]
                        
                        # Update V
                        for c in range(2):
                            for d in range(2):
                                coeff = beta[i][c][d]
                                if coeff != 0:
                                    global_row = c * half + u
                                    global_col = d * half + v
                                    global_idx = global_row * s + global_col
                                    V[global_idx][r] += coeff * V_s[local_idx][k]
                        
                        # Update W
                        for a in range(2):
                            for b in range(2):
                                coeff = gamma[i][a][b]
                                if coeff != 0:
                                    global_row = a * half + u
                                    global_col = b * half + v
                                    # W index is k*n + i, here k is col, i is row
                                    # So index = global_col * s + global_row
                                    global_idx = global_col * s + global_row
                                    W[global_idx][r] += coeff * W_s[local_idx][k]
                                    
        return U, V, W

    # Get decomposition for 4x4x4
    U_444, V_444, W_444 = get_strassen_decomp(4)
    rank_444 = len(U_444[0])
    
    # Assemble for 4x4x8
    rank = 2 * rank_444
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    for r in range(rank_444):
        # Copy U
        for i in range(16):
            U[i][r] = U_444[i][r]
            U[i][r + rank_444] = U_444[i][r]
            
        # Copy V
        # V_444 index j*4+k maps to global j*8+k for first half
        # and j*8+(4+k) for second half
        for j in range(4):
            for k in range(4):
                local_idx = j * 4 + k
                val = V_444[local_idx][r]
                V[j * 8 + k][r] = val
                V[j * 8 + 4 + k][r + rank_444] = val
                
        # Copy W
        # W_444 index k*4+i maps to global k*4+i for first half
        # and (4+k)*4+i for second half
        for k in range(4):
            for i in range(4):
                local_idx = k * 4 + i
                val = W_444[local_idx][r]
                W[k * 4 + i][r] = val
                W[(4 + k) * 4 + i][r + rank_444] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
