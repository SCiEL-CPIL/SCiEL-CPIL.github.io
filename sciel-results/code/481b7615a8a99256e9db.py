import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (Strassen) + 16 for 4x4x1 part (Schoolbook)
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # u, v, w vectors of length 4 (indices 0..3 for 2x2 matrix elements)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    v2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    w2 = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 1, 0, -1],# M2 -> C10, C11
        [0, 0, 1, 1], # M3 -> C01, C11
        [1, 1, 0, 0], # M4 -> C00, C10
        [-1, 0, 1, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    # Note: w2 indices correspond to C blocks:
    # 0: C00, 1: C10, 2: C01, 3: C11
    # Wait, my previous mapping:
    # W index k*2+i.
    # C00 (k=0,i=0) -> 0
    # C01 (k=1,i=0) -> 2
    # C10 (k=0,i=1) -> 1
    # C11 (k=1,i=1) -> 3
    # So w2 vectors should be in order [C00, C10, C01, C11].
    # Let's re-verify w2 values.
    # M1: C00(+1), C11(+1) -> [1, 0, 0, 1] (Correct)
    # M2: C10(+1), C11(-1) -> [0, 1, 0, -1] (Correct)
    # M3: C01(+1), C11(+1) -> [0, 0, 1, 1] (Correct)
    # M4: C00(+1), C10(+1) -> [1, 1, 0, 0] (Correct)
    # M5: C00(-1), C01(+1) -> [-1, 0, 1, 0] (Correct)
    # M6: C11(+1) -> [0, 0, 0, 1] (Correct)
    # M7: C00(+1) -> [1, 0, 0, 0] (Correct)
    
    # Coefficients for X_s and Y_s
    # X_s = sum cX[s][a][b] * A_ab
    # Y_s = sum cY[s][c][d] * B_cd
    # a,b,c,d in {0,1}
    
    cX = [
        [[1,0],[0,1]], # M1: A00 + A11
        [[0,0],[1,1]], # M2: A10 + A11
        [[1,0],[0,0]], # M3: A00
        [[0,0],[0,1]], # M4: A11
        [[1,1],[0,0]], # M5: A00 + A01
        [[0,0],[1,-1]],# M6: A10 - A00
        [[0,1],[0,-1]] # M7: A01 - A11
    ]
    cY = [
        [[1,0],[0,1]], # M1: B00 + B11
        [[1,0],[0,0]], # M2: B00
        [[0,0],[1,-1]],# M3: B10 - B11
        [[0,1],[-1,0]],# M4: B01 - B00
        [[0,0],[0,1]], # M5: B11
        [[1,0],[1,0]], # M6: B00 + B10
        [[0,1],[0,1]]  # M7: B11 + B01
    ]
    
    # Build 4x4x4 part
    col = 0
    for s in range(7):
        for t in range(7):
            # u2[s][t] is coeff for A local elem
            # v2[s][t] is coeff for B local elem
            # w2[s][t] is coeff for C local elem
            
            # Iterate over local indices
            for il in range(2):
                for jl in range(2):
                    # X_s local index il, jl
                    # Global A index: i = a*2 + il, j = b*2 + jl
                    # X_s[il, jl] = sum_{a,b} cX[s][a][b] * A[a,b][il, jl]
                    # So U_global[i*4+j, col] += u2[s][il*2+jl] * cX[s][a][b]
                    # Wait, u2[s] is a vector of length 4.
                    # u2[s][il*2+jl] is the coefficient for the t-th term.
                    u_val = u2[s][il * 2 + jl]
                    if u_val == 0: continue
                    for a in range(2):
                        for b in range(2):
                            cx_val = cX[s][a][b]
                            if cx_val == 0: continue
                            i_global = a * 2 + il
                            j_global = b * 2 + jl
                            U[i_global * 4 + j_global][col] += u_val * cx_val
            
            for jl in range(2):
                for kl in range(2):
                    # Y_s local index jl, kl
                    # Global B index: j = c*2 + jl, k = d*2 + kl
                    v_val = v2[s][jl * 2 + kl]
                    if v_val == 0: continue
                    for c in range(2):
                        for d in range(2):
                            cy_val = cY[s][c][d]
                            if cy_val == 0: continue
                            j_global = c * 2 + jl
                            k_global = d * 2 + kl
                            V[j_global * 4 + k_global][col] += v_val * cy_val
            
            for kl in range(2):
                for il in range(2):
                    # Z_s local index kl, il
                    # Global C index: k = c*2 + kl, i = a*2 + il
                    # w2[s] is vector [C00, C10, C01, C11]
                    # Local C index: k_loc, i_loc.
                    # C00: k=0, i=0 -> idx 0
                    # C10: k=0, i=1 -> idx 1
                    # C01: k=1, i=0 -> idx 2
                    # C11: k=1, i=1 -> idx 3
                    # So idx = k_loc * 2 + i_loc.
                    w_val = w2[s][kl * 2 + il]
                    if w_val == 0: continue
                    # Z_s is added to C blocks with coefficients from w2?
                    # No, w2[s] is the coefficient for the ENTIRE block Z_s.
                    # Wait.
                    # C_block = sum_s w2[s][block_idx] * Z_s.
                    # Z_s is a 2x2 matrix.
                    # So C_block[kl, il] += w2[s][block_idx] * Z_s[kl, il].
                    # But Z_s is computed as sum_t ...
                    # So for each t, we add w2[s][block_idx] * (term from Z_s).
                    # The term from Z_s at local (kl, il) is determined by w2_2x2?
                    # No, the 2x2x2 decomposition gives Z_s[il, kl]?
                    # My w2_2x2 vectors are for the 2x2x2 tensor.
                    # The 2x2x2 tensor outputs C_2x2.
                    # So w2_2x2[s][t] is the coefficient for the t-th term in the 2x2x2 multiplication.
                    # But here w2[s] is the coefficient for the block.
                    # I need to combine them.
                    # Global W index: k_global * 4 + i_global.
                    # k_global = c * 2 + kl, i_global = a * 2 + il.
                    # The contribution is w2_block[s][block_idx] * w2_2x2[s][t][kl*2+il]?
                    # No.
                    # The 2x2x2 decomposition is for ONE multiplication.
                    # Here we have 7 such multiplications.
                    # For each s, we use the 2x2x2 decomposition.
                    # The output of the 2x2x2 decomposition is Z_s.
                    # Z_s is then scaled by w2_block[s] and added to C.
                    # So the W factor for this term is:
                    # sum_{block} w2_block[s][block] * W_2x2_2x2[t][local_idx]
                    # where local_idx corresponds to the position in the 2x2 block.
                    
                    # Let's restructure.
                    # For each s, t:
                    # U part: sum_{a,b,il,jl} cX[s][a][b] * u2_2x2[s][t][il*2+jl] * delta(i,a*2+il, j,b*2+jl)
                    # V part: sum_{c,d,jl,kl} cY[s][c][d] * v2_2x2[s][t][jl*2+kl] * delta(j,c*2+jl, k,d*2+kl)
                    # W part: sum_{c,a,kl,il} w2_block[s][c*2+a] * w2_2x2[s][t][kl*2+il] * delta(k,c*2+kl, i,a*2+il)
                    
                    # I already filled U and V.
                    # Now fill W.
                    pass
            
            col += 1
    
    # Fill W for 4x4x4 part
    # Re-do loop for W
    col = 0
    for s in range(7):
        for t in range(7):
            for c in range(2):
                for a in range(2):
                    block_idx = c * 2 + a
                    w_block_val = w2[s][block_idx]
                    if w_block_val == 0: continue
                    for kl in range(2):
                        for il in range(2):
                            w_2x2_val = w2_2x2_get(s, t, kl, il) # Need this function
                            if w_2x2_val == 0: continue
                            k_global = c * 2 + kl
                            i_global = a * 2 + il
                            W[k_global * 4 + i_global][col] += w_block_val * w_2x2_val
            col += 1
            
    # Add 4x4x1 part
    # Schoolbook
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# Helper to get w2_2x2 value
def w2_2x2_get(s, t, kl, il):
    # w2_2x2 is the w vector from the 2x2x2 decomposition
    # I defined w2 above as w2_2x2? No, I defined w2 as block coeffs.
    # I need the w vectors from the 2x2x2 Strassen.
    # Let's name them w_2x2.
    # The w vectors for 2x2x2 Strassen are the same as the w vectors in the 2x2x2 decomposition.
    # I used w2 in the code above for block coeffs. I should rename.
    pass
