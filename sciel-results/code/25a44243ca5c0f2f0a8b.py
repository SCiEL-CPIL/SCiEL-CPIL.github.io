"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 98 decomposition using Strassen for 4x4x4 blocks
    # M_{4,4,8} = M_{4,4,4} (L) + M_{4,4,4} (R)
    # Each M_{4,4,4} uses 7 Strassen steps of 2x2x2 (rank 7)
    # Total 7 * 7 * 2 = 98? No.
    # M_{4,4,4} rank 49.
    # M_{4,4,8} rank 98.
    
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (4x4)
    # Blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1)
    # A[i,j]
    
    # Indices for B (4x8)
    # Blocks for L (cols 0-3): 11, 12, 21, 22
    # Blocks for R (cols 4-7): 11, 12, 21, 22
    # B[i,j]
    
    # Indices for C (4x8)
    # Blocks for L: 11, 12, 21, 22
    # Blocks for R: 11, 12, 21, 22
    # C[i,j]
    
    # Strassen sums for A (2x2 blocks of 2x2 matrices)
    # S1 = A11 + A22
    # S2 = A21 + A22
    # S3 = A11
    # S4 = A22
    # S5 = A11 + A12
    # S6 = A21 - A11
    # S7 = A12 - A22
    
    # Strassen B-combos for L and R
    # For k=1: X = B11 + B22
    # For k=2: X = B11
    # For k=3: X = B12 - B22
    # For k=4: X = B21 - B11
    # For k=5: X = B22
    # For k=6: X = B11 + B12
    # For k=7: X = B21 + B22
    
    # Inner Strassen for 2x2x2 (S * X)
    # m1 = (s00+s11)(x00+x11)
    # m2 = (s10+s11)x00
    # m3 = s00(x01-x11)
    # m4 = s11(x10-x00)
    # m5 = (s00+s01)x11
    # m6 = (s10-s00)(x00+x01)
    # m7 = (s01-s11)(x10+x11)
    
    # Output Y = S*X
    # y00 = m1 + m4 - m5 + m7
    # y01 = m3 + m5
    # y10 = m2 + m4
    # y11 = m1 - m2 + m3 + m6
    
    # Reconstruction of C from Ys
    # C11 = Y1 + Y4 - Y5 + Y7
    # C12 = Y3 + Y5
    # C21 = Y2 + Y4
    # C22 = Y1 - Y2 + Y3 + Y6
    
    # Helper to get A indices for block (bi, bj)
    # bi, bj in {0,1}
    # A block (bi,bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    def a_idx(bi, bj, ri, cj):
        return (2*bi + ri) * 4 + (2*bj + cj)
        
    def b_idx_L(bi, bj, ri, cj):
        return (2*bi + ri) * 8 + (2*bj + cj)
        
    def b_idx_R(bi, bj, ri, cj):
        return (2*bi + ri) * 8 + (4 + 2*bj + cj)
        
    def c_idx_L(bi, bj, ri, cj):
        return (2*bi + ri) * 8 + (2*bj + cj)
        
    def c_idx_R(bi, bj, ri, cj):
        return (2*bi + ri) * 8 + (4 + 2*bj + cj)

    # Define A factors for S1..S7
    # Each Sk is a 2x2 matrix.
    # We need the coefficients for each entry of Sk in terms of A.
    # S1: A11 + A22
    # S2: A21 + A22
    # S3: A11
    # S4: A22
    # S5: A11 + A12
    # S6: A21 - A11
    # S7: A12 - A22
    
    # For inner Strassen, we need s00, s01, s10, s11.
    # s00 is entry (0,0) of Sk.
    
    # Let's precompute the A-coefficients for each Sk and each inner-Strassen factor.
    # Inner Strassen factors:
    # f1_s = s00 + s11
    # f2_s = s10 + s11
    # f3_s = s00
    # f4_s = s11
    # f5_s = s00 + s01
    # f6_s = s10 - s00
    # f7_s = s01 - s11
    
    # For each Sk, we define a function or dict mapping (ri, cj) to list of (a_index, coeff)
    # Actually, we can just iterate.
    
    col = 0
    for k in range(1, 8):
        for side in ['L', 'R']:
            for t in range(1, 8):
                # Determine A factor for this t
                # t=1: s00+s11
                # t=2: s10+s11
                # t=3: s00
                # t=4: s11
                # t=5: s00+s01
                # t=6: s10-s00
                # t=7: s01-s11
                
                # Determine B factor for this t
                # t=1: x00+x11
                # t=2: x00
                # t=3: x01-x11
                # t=4: x10-x00
                # t=5: x11
                # t=6: x00+x01
                # t=7: x10+x11
                
                # We need to add to U, V, W
                
                # 1. U (A part)
                # Sk depends on k.
                # We need the specific A indices involved in the t-th factor of Sk.
                
                # Let's build the A factor explicitly.
                # Sk is 2x2.
                # S1: [[A11_00+A22_00, A11_01+A22_01], [A11_10+A22_10, A11_11+A22_11]]
                
                # This is getting complex to hardcode.
                # Let's use a helper to get the A-block entries.
                
                def get_A_block(bi, bj):
                    # Returns 2x2 list of (index, coeff)
                    # bi, bj in 0,1
                    res = [[], [], [], []] # ri, cj
                    for ri in range(2):
                        for cj in range(2):
                            idx = a_idx(bi, bj, ri, cj)
                            res[ri][cj] = (idx, 1)
                    return res

                # Define Sk blocks
                if k == 1:
                    # A11 + A22
                    blocks = [(0,0,1), (1,1,1)]
                elif k == 2:
                    # A21 + A22
                    blocks = [(1,0,1), (1,1,1)]
                elif k == 3:
                    # A11
                    blocks = [(0,0,1)]
                elif k == 4:
                    # A22
                    blocks = [(1,1,1)]
                elif k == 5:
                    # A11 + A12
                    blocks = [(0,0,1), (0,1,1)]
                elif k == 6:
                    # A21 - A11
                    blocks = [(1,0,1), (0,0,-1)]
                elif k == 7:
                    # A12 - A22
                    blocks = [(0,1,1), (1,1,-1)]
                
                # Build Sk 2x2 matrix of coefficients
                # Sk[ri][cj] = sum of coeff * A[bi,bj,ri,cj]
                Sk = [[{}, {}, {}, {}], [{}, {}, {}, {}]] # ri, cj -> {a_idx: coeff}
                for bi, bj, coeff in blocks:
                    Ab = get_A_block(bi, bj)
                    for ri in range(2):
                        for cj in range(2):
                            idx, c = Ab[ri][cj]
                            Sk[ri][cj][idx] = Sk[ri][cj].get(idx, 0) + coeff * c

                # Now apply inner Strassen factor t to Sk
                # t=1: s00+s11 -> Sk[0][0] + Sk[1][1]
                # t=2: s10+s11 -> Sk[1][0] + Sk[1][1]
                # t=3: s00 -> Sk[0][0]
                # t=4: s11 -> Sk[1][1]
                # t=5: s00+s01 -> Sk[0][0] + Sk[0][1]
                # t=6: s10-s00 -> Sk[1][0] - Sk[0][0]
                # t=7: s01-s11 -> Sk[0][1] - Sk[1][1]
                
                def add_dict(d1, d2, sign=1):
                    res = {}
                    for idx, c in d1.items():
                        res[idx] = res.get(idx, 0) + c
                    for idx, c in d2.items():
                        res[idx] = res.get(idx, 0) + sign * c
                    return res

                if t == 1:
                    A_factor = add_dict(Sk[0][0], Sk[1][1])
                elif t == 2:
                    A_factor = add_dict(Sk[1][0], Sk[1][1])
                elif t == 3:
                    A_factor = Sk[0][0]
                elif t == 4:
                    A_factor = Sk[1][1]
                elif t == 5:
                    A_factor = add_dict(Sk[0][0], Sk[0][1])
                elif t == 6:
                    A_factor = add_dict(Sk[1][0], Sk[0][0], -1)
                elif t == 7:
                    A_factor = add_dict(Sk[0][1], Sk[1][1], -1)

                # 2. V (B part)
                # X_k,side
                # k=1: B11 + B22
                # k=2: B11
                # k=3: B12 - B22
                # k=4: B21 - B11
                # k=5: B22
                # k=6: B11 + B12
                # k=7: B21 + B22
                
                def get_B_block(side, bi, bj):
                    res = [[], [], [], []]
                    for ri in range(2):
                        for cj in range(2):
                            if side == 'L':
                                idx = b_idx_L(bi, bj, ri, cj)
                            else:
                                idx = b_idx_R(bi, bj, ri, cj)
                            res[ri][cj] = (idx, 1)
                    return res

                if k == 1:
                    B_blocks = [(0,0,1), (1,1,1)]
                elif k == 2:
                    B_blocks = [(0,0,1)]
                elif k == 3:
                    B_blocks = [(0,1,1), (1,1,-1)]
                elif k == 4:
                    B_blocks = [(1,0,1), (0,0,-1)]
                elif k == 5:
                    B_blocks = [(1,1,1)]
                elif k == 6:
                    B_blocks = [(0,0,1), (0,1,1)]
                elif k == 7:
                    B_blocks = [(1,0,1), (1,1,1)]
                
                Xk = [[{}, {}, {}, {}], [{}, {}, {}, {}]]
                for bi, bj, coeff in B_blocks:
                    Bb = get_B_block(side, bi, bj)
                    for ri in range(2):
                        for cj in range(2):
                            idx, c = Bb[ri][cj]
                            Xk[ri][cj][idx] = Xk[ri][cj].get(idx, 0) + coeff * c

                # Inner Strassen factor t for X
                # t=1: x00+x11
                # t=2: x00
                # t=3: x01-x11
                # t=4: x10-x00
                # t=5: x11
                # t=6: x00+x01
                # t=7: x10+x11
                
                if t == 1:
                    B_factor = add_dict(Xk[0][0], Xk[1][1])
                elif t == 2:
                    B_factor = Xk[0][0]
                elif t == 3:
                    B_factor = add_dict(Xk[0][1], Xk[1][1], -1)
                elif t == 4:
                    B_factor = add_dict(Xk[1][0], Xk[0][0], -1)
                elif t == 5:
                    B_factor = Xk[1][1]
                elif t == 6:
                    B_factor = add_dict(Xk[0][0], Xk[0][1])
                elif t == 7:
                    B_factor = add_dict(Xk[1][0], Xk[1][1])

                # 3. W (C part)
                # Y_t is the result of the t-th Strassen product.
                # Y_t is a 2x2 matrix.
                # We need to add Y_t to the appropriate C blocks.
                
                # Y_t entries:
                # y00 = m1 + m4 - m5 + m7
                # y01 = m3 + m5
                # y10 = m2 + m4
                # y11 = m1 - m2 + m3 + m6
                
                # This Y_t contributes to C_L or C_R.
                # C block reconstruction:
                # C11 += Y1 + Y4 - Y5 + Y7
                # C12 += Y3 + Y5
                # C21 += Y2 + Y4
                # C22 += Y1 - Y2 + Y3 + Y6
                
                # So for this specific t, we know which C blocks it contributes to and with what sign.
                
                # t=1: C11 (+1), C22 (+1)
                # t=2: C21 (+1), C22 (-1)
                # t=3: C12 (+1), C22 (+1)
                # t=4: C11 (+1), C21 (+1)
                # t=5: C11 (-1), C12 (+1)
                # t=6: C22 (+1)
                # t=7: C11 (+1)
                
                # We also need to map Y_t[ri][cj] to C indices.
                # Y_t is 2x2.
                # Y_t[0][0] -> C_block[0][0,0]
                # Y_t[0][1] -> C_block[0][0,1]
                # Y_t[1][0] -> C_block[0][1,0]
                # Y_t[1][1] -> C_block[0][1,1]
                
                # C_block is one of C11, C12, C21, C22.
                
                contributions = {}
                # contributions: { (c_bi, c_bj, ri, cj): coeff }
                
                if t == 1:
                    contributions[(0,0)] = 1 # C11
                    contributions[(1,1)] = 1 # C22
                elif t == 2:
                    contributions[(1,0)] = 1 # C21
                    contributions[(1,1)] = -1 # C22
                elif t == 3:
                    contributions[(0,1)] = 1 # C12
                    contributions[(1,1)] = 1 # C22
                elif t == 4:
                    contributions[(0,0)] = 1 # C11
                    contributions[(1,0)] = 1 # C21
                elif t == 5:
                    contributions[(0,0)] = -1 # C11
                    contributions[(0,1)] = 1 # C12
                elif t == 6:
                    contributions[(1,1)] = 1 # C22
                elif t == 7:
                    contributions[(0,0)] = 1 # C11

                # Now populate W
                for (c_bi, c_bj), coeff in contributions.items():
                    for ri in range(2):
                        for cj in range(2):
                            if side == 'L':
                                idx = c_idx_L(c_bi, c_bj, ri, cj)
                            else:
                                idx = c_idx_R(c_bi, c_bj, ri, cj)
                            W[idx][col] += coeff

                # Populate U and V
                for idx, coeff in A_factor.items():
                    U[idx][col] += coeff
                for idx, coeff in B_factor.items():
                    V[idx][col] += coeff

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
