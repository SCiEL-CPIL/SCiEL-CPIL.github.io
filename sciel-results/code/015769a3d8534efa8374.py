import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # We decompose M_{2,4,6} into 6 blocks of M_{2,2,2}
    # A is 2x4, split into A_L (2x2) and A_R (2x2)
    # B is 4x6, split into B_1 (4x2), B_2 (4x2), B_3 (4x2)
    # Each B_j is split into top 2x2 and bottom 2x2.
    # The product is sum over i in {L,R} and j in {1,2,3} of A_i * B_{j,i}
    # where B_{j,i} is the appropriate 2x2 submatrix of B_j.
    # Specifically:
    # C = A_L (B_1_top + B_2_top + B_3_top) + A_R (B_1_bot + B_2_bot + B_3_bot)
    # This is not quite right. The multiplication is C = A B.
    # A = [A11 A12; A21 A22] (2x4)
    # B = [B11 B12; B21 B22] where B11 is 2x6, B21 is 2x6.
    # Actually, let's stick to the block structure used in the seed.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [B_1, B_2, B_3] where B_j are 4x2.
    # B_j = [B_{j,1}; B_{j,2}] where B_{j,1}, B_{j,2} are 2x2.
    # C = A_L B_{1,1} + A_L B_{2,1} + A_L B_{3,1} + A_R B_{1,2} + A_R B_{2,2} + A_R B_{3,2}
    # Wait, A is 2x4. B is 4x6.
    # A = [A_{1,1} A_{1,2}; A_{2,1} A_{2,2}] is not standard.
    # Let A = [A_L, A_R] where A_L is cols 0-1, A_R is cols 2-3. Both 2x2.
    # Let B be split into 3 blocks of 4x2: B_1 (cols 0-1), B_2 (cols 2-3), B_3 (cols 4-5).
    # B_j = [B_{j,top}; B_{j,bot}] where B_{j,top} is rows 0-1 (2x2), B_{j,bot} is rows 2-3 (2x2).
    # The product C = A B has entries C_{ik} = sum_j A_{ij} B_{jk}.
    # C_{i, k} where k is in block b (0,1,2) and local k_loc.
    # j ranges 0..3.
    # If j is in 0,1 (A_L part), we use A_L. B_{j,k} is in B_{b,top}.
    # If j is in 2,3 (A_R part), we use A_R. B_{j,k} is in B_{b,bot}.
    # So C_{i, b*2 + k_loc} = (A_L * (B_{b,top}[:, k_loc]))_i + (A_R * (B_{b,bot}[:, k_loc]))_i.
    # This is equivalent to:
    # C = [A_L B_{0,top} + A_R B_{0,bot}, A_L B_{1,top} + A_R B_{1,bot}, A_L B_{2,top} + A_R B_{2,bot}]
    # Each term is a product of a 2x2 matrix and a 2x2 matrix.
    # There are 3 such terms. Each term involves two 2x2x2 multiplications.
    # Total 6 2x2x2 multiplications.
    # Rank = 6 * 7 = 42.

    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42

    # Strassen coefficients for 2x2x2
    # Inputs: X (2x2), Y (2x2). Output: Z (2x2).
    # U_X[s] is coefficients for X. V_Y[s] for Y. W_Z[s] for Z.
    # X indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Y indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Z indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6

    U_strassen = [
        [(0, 1), (3, 1)],       # P1: X00, X11
        [(2, 1), (3, 1)],       # P2: X10, X11
        [(0, 1)],               # P3: X00
        [(3, 1)],               # P4: X11
        [(0, 1), (1, 1)],       # P5: X00, X01
        [(2, 1), (0, -1)],      # P6: X10, -X00
        [(1, 1), (3, -1)]       # P7: X01, -X11
    ]
    
    V_strassen = [
        [(0, 1), (3, 1)],       # P1: Y00, Y11
        [(0, 1)],               # P2: Y00
        [(1, 1), (3, -1)],      # P3: Y01, -Y11
        [(2, 1), (0, -1)],      # P4: Y10, -Y00
        [(3, 1)],               # P5: Y11
        [(0, 1), (1, 1)],       # P6: Y00, Y01
        [(2, 1), (3, 1)]        # P7: Y10, Y11
    ]
    
    W_strassen = [
        [(0, 1), (3, 1)],       # P1: Z00, Z11
        [(2, 1), (3, -1)],      # P2: Z10, -Z11
        [(1, 1), (3, 1)],       # P3: Z01, Z11
        [(0, 1), (2, 1)],       # P4: Z00, Z10
        [(0, -1), (1, 1)],      # P5: -Z00, Z01
        [(3, 1)],               # P6: Z11
        [(0, 1)]                # P7: Z00
    ]

    col = 0
    # 3 blocks of B columns (b=0,1,2)
    # 2 blocks of A columns (a=0,1) where a=0 is A_L (cols 0-1 of A, rows 0-1 of B block), a=1 is A_R (cols 2-3 of A, rows 2-3 of B block)
    for b in range(3):
        for a in range(2):
            for s in range(7):
                # Update U (A matrix)
                # A is 2x4. Index i*m + j.
                # If a=0, we use A_L which is A[:, 0:2]. Local j is 0,1. Global j is j_loc.
                # If a=1, we use A_R which is A[:, 2:4]. Local j is 0,1. Global j is 2 + j_loc.
                for (idx, coeff) in U_strassen[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V (B matrix)
                # B is 4x6. Index j*p + k.
                # For block b, k is in b*2 .. b*2+1.
                # If a=0, we use B_{b,top} which is B[0:2, b*2:b*2+2]. Local j is 0,1. Global j is j_loc.
                # If a=1, we use B_{b,bot} which is B[2:4, b*2:b*2+2]. Local j is 0,1. Global j is 2 + j_loc.
                for (idx, coeff) in V_strassen[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W (Output C)
                # C is 2x6. Index k*n + i.
                # k is in b*2 .. b*2+1.
                # i is 0,1.
                for (idx, coeff) in W_strassen[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
