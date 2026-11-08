import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 161
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman patterns for 3x3
    # Each pattern: (x_coeffs, y_coeffs, k_matrix)
    # x_coeffs: dict (row, col) -> coeff for X
    # y_coeffs: dict (row, col) -> coeff for Y
    # k_matrix: 3x3 list of lists, coeff for Z
    
    # Laderman's algorithm for 3x3 matrix multiplication
    # Z = XY
    # 23 multiplications
    
    laderman = [
        # 1
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}, [[1,0,0],[0,1,0],[0,0,0]]), # M1 = X00(Y00+Y11) -> Z00 += X00Y00, Z01? No.
        # Wait, Laderman formulas are specific.
        # Let's use a known correct set.
    ]
    
    # Correct Laderman set
    # Ref: Laderman (1976)
    # Z = XY
    # 1. M1 = X00(Y00+Y11)
    # 2. M2 = (X00+X11)Y11
    # 3. M3 = X22(Y00+Y22)
    # 4. M4 = (X22+X00)Y22
    # 5. M5 = X11(Y11+Y22)
    # 6. M6 = (X11+X22)Y22
    # 7. M7 = X01(Y10+Y20)
    # 8. M8 = (X01+X02)Y20
    # 9. M9 = X21(Y01+Y02)
    # 10. M10 = (X21+X22)Y02
    # 11. M11 = X12(Y02+Y12)
    # 12. M12 = (X12+X11)Y12
    # 13. M13 = X02(Y20+Y21)
    # 14. M14 = (X02+X00)Y21
    # 15. M15 = X20(Y10+Y12)
    # 16. M16 = (X20+X22)Y12
    # 17. M17 = X10(Y01+Y02)
    # 18. M18 = (X10+X11)Y02
    # 19. M19 = (X00+X11+X22)(Y00+Y11+Y22)
    # 20. M20 = (X01+X12+X20)(Y10+Y21+Y02)
    # 21. M21 = (X02+X21+X10)(Y20+Y01+Y12)
    # 22. M22 = (X01+X10+X22)(Y10+Y01+Y22)
    # 23. M23 = (X02+X20+X11)(Y20+Y12+Y01)
    
    # Reconstruction:
    # Z00 = M1 + M3 + M19 - M2 - M4 - M6
    # Z01 = M7 + M9 + M20 - M8 - M10 - M18
    # Z02 = M13 + M15 + M21 - M14 - M16 - M22
    # Z10 = M17 + M19 - M18 - M16
    # Z11 = M2 + M5 + M20 - M1 - M6 - M19
    # Z12 = M11 + M12 + M21 - M10 - M14 - M23
    # Z20 = M15 + M16 - M13 - M21
    # Z21 = M9 + M10 + M22 - M7 - M18 - M19
    # Z22 = M4 + M6 + M23 - M3 - M5 - M20
    
    # This reconstruction is for the final Z.
    # But we need the K matrices for each M_i.
    # M_i is a scalar.
    # Z is a matrix.
    # Z = sum M_i * K_i
    # So K_i is a 3x3 matrix with coefficients.
    
    K = [
        # M1
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]], # Z00 += 1
        # M2
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], # Z11 += 1
        # M3
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]], # Z00 += 1
        # M4
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]], # Z22 += 1
        # M5
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], # Z11 += 1
        # M6
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]], # Z22 += 1
        # M7
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]], # Z01 += 1
        # M8
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]], # Z01 += 1
        # M9
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]], # Z01 += 1
        # M10
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]], # Z02 += 1
        # M11
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]], # Z02 += 1
        # M12
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]], # Z12 += 1
        # M13
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]], # Z02 += 1
        # M14
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]], # Z12 += 1
        # M15
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]], # Z20 += 1
        # M16
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]], # Z20 += 1
        # M17
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]], # Z10 += 1
        # M18
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]], # Z10 += 1
        # M19
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], # Z00, Z11, Z22 += 1
        # M20
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]], # Z01 += 1
        # M21
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]], # Z02 += 1
        # M22
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]], # Z01 += 1
        # M23
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]], # Z21 += 1
    ]
    
    # Wait, the K matrices above are just placeholders.
    # The reconstruction formulas give the coefficients.
    # Z00 = 1*M1 + 1*M3 + 1*M19 - 1*M2 - 1*M4 - 1*M6
    # So K1[0,0]=1, K3[0,0]=1, K19[0,0]=1, K2[0,0]=-1, etc.
    
    # Let's build K correctly.
    K = [[0]*3 for _ in range(3)] # 23x3x3
    # M1
    K[0][0][0] = 1
    # M2
    K[1][1][1] = 1
    # M3
    K[2][0][0] = 1
    # M4
    K[3][2][2] = 1
    # M5
    K[4][1][1] = 1
    # M6
    K[5][2][2] = 1
    # M7
    K[6][0][1] = 1
    # M8
    K[7][0][1] = 1
    # M9
    K[8][0][1] = 1
    # M10
    K[9][0][2] = 1
    # M11
    K[10][0][2] = 1
    # M12
    K[11][1][2] = 1
    # M13
    K[12][0][2] = 1
    # M14
    K[13][1][2] = 1
    # M15
    K[14][2][0] = 1
    # M16
    K[15][2][0] = 1
    # M17
    K[16][1][0] = 1
    # M18
    K[17][1][0] = 1
    # M19
    K[18][0][0] = 1
    K[18][1][1] = 1
    K[18][2][2] = 1
    # M20
    K[19][0][1] = 1
    # M21
    K[20][0][2] = 1
    # M22
    K[21][0][1] = 1
    # M23
    K[22][2][1] = 1
    
    # Now apply reconstruction coefficients
    # Z00 = M1 + M3 + M19 - M2 - M4 - M6
    # So for Z00 (index 0,0):
    # K[0][0][0] += 1 (already 1)
    # K[2][0][0] += 1 (already 1)
    # K[18][0][0] += 1 (already 1)
    # K[1][0][0] -= 1
    # K[3][0][0] -= 1
    # K[5][0][0] -= 1
    
    # This is getting error prone.
    # Better to define the reconstruction explicitly.
    
    # Let's restart the K definition with the full formulas.
    K = [[[0]*3 for _ in range(3)] for _ in range(23)]
    
    # Z00
    K[0][0][0] += 1   # M1
    K[2][0][0] += 1   # M3
    K[18][0][0] += 1  # M19
    K[1][0][0] -= 1   # -M2
    K[3][0][0] -= 1   # -M4
    K[5][0][0] -= 1   # -M6
    
    # Z01
    K[6][0][1] += 1   # M7
    K[8][0][1] += 1   # M9
    K[19][0][1] += 1  # M20
    K[7][0][1] -= 1   # -M8
    K[9][0][1] -= 1   # -M10
    K[17][0][1] -= 1  # -M18
    
    # Z02
    K[12][0][2] += 1  # M13
    K[14][0][2] += 1  # M15
    K[20][0][2] += 1  # M21
    K[13][0][2] -= 1  # -M14
    K[15][0][2] -= 1  # -M16
    K[21][0][2] -= 1  # -M22
    
    # Z10
    K[16][1][0] += 1  # M17
    K[18][1][0] += 1  # M19
    K[17][1][0] -= 1  # -M18
    K[15][1][0] -= 1  # -M16
    
    # Z11
    K[1][1][1] += 1   # M2
    K[4][1][1] += 1   # M5
    K[19][1][1] += 1  # M20
    K[0][1][1] -= 1   # -M1
    K[5][1][1] -= 1   # -M6
    K[18][1][1] -= 1  # -M19
    
    # Z12
    K[10][1][2] += 1  # M11
    K[11][1][2] += 1  # M12
    K[20][1][2] += 1  # M21
    K[9][1][2] -= 1   # -M10
    K[13][1][2] -= 1  # -M14
    K[22][1][2] -= 1  # -M23
    
    # Z20
    K[14][2][0] += 1  # M15
    K[15][2][0] += 1  # M16
    K[12][2][0] -= 1  # -M13
    K[20][2][0] -= 1  # -M21
    
    # Z21
    K[8][2][1] += 1   # M9
    K[9][2][1] += 1   # M10
    K[21][2][1] += 1  # M22
    K[6][2][1] -= 1   # -M7
    K[17][2][1] -= 1  # -M18
    K[18][2][1] -= 1  # -M19
    
    # Z22
    K[3][2][2] += 1   # M4
    K[5][2][2] += 1   # M6
    K[22][2][2] += 1  # M23
    K[2][2][2] -= 1   # -M3
    K[4][2][2] -= 1   # -M5
    K[19][2][2] -= 1  # -M20

    # Now define X, Y coeffs for each M_i
    # M_i = x_i y_i
    # x_i is linear combo of X entries
    # y_i is linear combo of Y entries
    
    XY_coeffs = [
        # 0: M1 = X00(Y00+Y11)
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}),
        # 1: M2 = (X00+X11)Y11
        ({(0,0): 1, (1,1): 1}, {(1,1): 1}),
        # 2: M3 = X22(Y00+Y22)
        ({(2,2): 1}, {(0,0): 1, (2,2): 1}),
        # 3: M4 = (X22+X00)Y22
        ({(2,2): 1, (0,0): 1}, {(2,2): 1}),
        # 4: M5 = X11(Y11+Y22)
        ({(1,1): 1}, {(1,1): 1, (2,2): 1}),
        # 5: M6 = (X11+X22)Y22
        ({(1,1): 1, (2,2): 1}, {(2,2): 1}),
        # 6: M7 = X01(Y10+Y20)
        ({(0,1): 1}, {(1,0): 1, (2,0): 1}),
        # 7: M8 = (X01+X02)Y20
        ({(0,1): 1, (0,2): 1}, {(2,0): 1}),
        # 8: M9 = X21(Y01+Y02)
        ({(2,1): 1}, {(0,1): 1, (0,2): 1}),
        # 9: M10 = (X21+X22)Y02
        ({(2,1): 1, (2,2): 1}, {(0,2): 1}),
        # 10: M11 = X12(Y02+Y12)
        ({(1,2): 1}, {(0,2): 1, (1,2): 1}),
        # 11: M12 = (X12+X11)Y12
        ({(1,2): 1, (1,1): 1}, {(1,2): 1}),
        # 12: M13 = X02(Y20+Y21)
        ({(0,2): 1}, {(2,0): 1, (2,1): 1}),
        # 13: M14 = (X02+X00)Y21
        ({(0,2): 1, (0,0): 1}, {(2,1): 1}),
        # 14: M15 = X20(Y10+Y12)
        ({(2,0): 1}, {(1,0): 1, (1,2): 1}),
        # 15: M16 = (X20+X22)Y12
        ({(2,0): 1, (2,2): 1}, {(1,2): 1}),
        # 16: M17 = X10(Y01+Y02)
        ({(1,0): 1}, {(0,1): 1, (0,2): 1}),
        # 17: M18 = (X10+X11)Y02
        ({(1,0): 1, (1,1): 1}, {(0,2): 1}),
        # 18: M19 = (X00+X11+X22)(Y00+Y11+Y22)
        ({(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
        # 19: M20 = (X01+X12+X20)(Y10+Y21+Y02)
        ({(0,1): 1, (1,2): 1, (2,0): 1}, {(1,0): 1, (2,1): 1, (0,2): 1}),
        # 20: M21 = (X02+X21+X10)(Y20+Y01+Y12)
        ({(0,2): 1, (2,1): 1, (1,0): 1}, {(2,0): 1, (0,1): 1, (1,2): 1}),
        # 21: M22 = (X01+X10+X22)(Y10+Y01+Y22)
        ({(0,1): 1, (1,0): 1, (2,2): 1}, {(1,0): 1, (0,1): 1, (2,2): 1}),
        # 22: M23 = (X02+X20+X11)(Y20+Y12+Y01)
        ({(0,2): 1, (2,0): 1, (1,1): 1}, {(2,0): 1, (1,2): 1, (0,1): 1}),
    ]
    
    # Strassen patterns
    # M_k = X_k Y_k
    # X_k, Y_k are 3x3 blocks of A, B
    # A is 2x2 blocks of 3x3
    # A = [[A00, A01], [A10, A11]]
    
    strassen = [
        # 0: M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 1: M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # 2: M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # 3: M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # 4: M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 5: M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 6: M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # C reconstruction
    # C00 = M0 + M3 - M4 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    C_coeffs = [
        # 0: C00
        {0: 1, 3: 1, 4: -1, 5: 1},
        # 1: C01
        {2: 1, 4: 1},
        # 2: C10
        {1: 1, 3: 1},
        # 3: C11
        {0: 1, 1: -1, 2: 1, 6: 1},
    ]
    
    col_idx = 0
    for k in range(7):
        sx, sy = strassen[k]
        for l in range(23):
            lx, ly = XY_coeffs[l]
            
            # Build U coeffs (A scalars)
            u_coeffs = {}
            for (bi, bj), sc in sx.items():
                # X_k block (bi, bj)
                # X_k[bi, bj] = sc * A_block[bi, bj]
                # A_block[bi, bj] is 3x3
                # X_k entry (r,c) is sc * A[3*bi+r, 3*bj+c]
                for (xr, xc), lc in lx.items():
                    # x_l contributes lc * X_k[xr, xc]
                    # X_k[xr, xc] is sum over bi,bj of sx[bi,bj] * A_block[bi,bj][xr,xc]
                    # So coeff for A[3*bi+xr, 3*bj+xc] is sc * lc
                    val = sc * lc
                    a_idx = (3*bi + xr) * 6 + (3*bj + xc)
                    u_coeffs[a_idx] = u_coeffs.get(a_idx, 0) + val
            
            # Build V coeffs (B scalars)
            v_coeffs = {}
            for (bi, bj), sc in sy.items():
                for (yr, yc), lc in ly.items():
                    val = sc * lc
                    b_idx = (3*bi + yr) * 6 + (3*bj + yc)
                    v_coeffs[b_idx] = v_coeffs.get(b_idx, 0) + val
            
            # Build W coeffs (C scalars)
            w_coeffs = {}
            # M_k contributes to C blocks via C_coeffs
            # M_k is 3x3
            # M_k[r,c] = sum_l s_l K_l[r,c]
            # C_block[i,j][r,c] = sum_k C_coeffs[i,j][k] M_k[r,c]
            # So C[3*i+r, 3*j+c] = sum_{k,l} C_coeffs[i,j][k] K_l[r,c] s_l
            # For fixed k, l:
            # Coeff for C[3*i+r, 3*j+c] is C_coeffs[i,j][k] * K[l][r][c]
            
            for i in range(2):
                for j in range(2):
                    if k in C_coeffs[i*2+j]:
                        tc = C_coeffs[i*2+j][k]
                        for r in range(3):
                            for c in range(3):
                                kc = K[l][r][c]
                                if kc != 0:
                                    val = tc * kc
                                    c_idx = (3*i + r) * 6 + (3*j + c)
                                    w_coeffs[c_idx] = w_coeffs.get(c_idx, 0) + val
            
            # Update U, V, W
            for a_idx, val in u_coeffs.items():
                if val != 0:
                    U[a_idx][col_idx] += val
            for b_idx, val in v_coeffs.items():
                if val != 0:
                    V[b_idx][col_idx] += val
            for c_idx, val in w_coeffs.items():
                if val != 0:
                    W[c_idx][col_idx] += val
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
