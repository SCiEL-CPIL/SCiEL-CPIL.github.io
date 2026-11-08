"""Program for rank-minimal exact tensor decomposition of 2x4 by 4x4 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The approach splits the 2x4 matrix A into two 2x2 blocks A0, A1.
    The 4x4 matrix B is split into four 2x2 blocks B00, B01, B10, B11.
    The product C = AB is a 2x4 matrix, split into two 2x2 blocks C0, C1.
    C0 = A0*B00 + A1*B10
    C1 = A0*B01 + A1*B11
    
    We use the identity:
    X*Y + X'*Y' = 1/2 * [ (X+X')*(Y+Y') - (X-X')*(Y-Y') ]
    This allows us to compute the sum of two 2x2 products using two 2x2 products.
    However, we have two such sums:
    C0 = A0*B00 + A1*B10
    C1 = A0*B01 + A1*B11
    
    Let A+ = A0 + A1, A- = A0 - A1.
    For C0:
    C0 = 1/2 * [ (A0+A1)*(B00+B10) - (A0-A1)*(B00-B10) ]
    Let P1 = A+ * (B00+B10)
    Let P2 = A- * (B00-B10)
    C0 = 1/2 * (P1 - P2)
    
    For C1:
    C1 = 1/2 * [ (A0+A1)*(B01+B11) - (A0-A1)*(B01-B11) ]
    Let P3 = A+ * (B01+B11)
    Let P4 = A- * (B01-B11)
    C1 = 1/2 * (P3 - P4)
    
    We need to compute 4 matrix products of size 2x2:
    P1, P2, P3, P4.
    Each 2x2 product has rank 7 (Strassen).
    Total rank = 4 * 7 = 28.
    
    Strassen's algorithm for 2x2 matrix multiplication:
    Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    M1 = (x00+x11)(y00+y11)
    M2 = (x10+x11)y00
    M3 = x00(y01-y11)
    M4 = x11(y10-y00)
    M5 = (x00+x01)y11
    M6 = (x10-x00)(y00+y01)
    M7 = (x01-x11)(y10+y11)
    
    Z = X*Y
    z00 = M1 - M3 + M4 + M5
    z01 = M3 + M5 - M7
    z10 = M2 + M4 + M7
    z11 = M1 - M2 + M6 + M7
    
    We will construct the U, V, W matrices for the 28 bilinear forms.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Indices for A (2x4):
    # A0 = [[A00, A01], [A02, A03]] -> indices 0,1,2,3
    # A1 = [[A10, A11], [A12, A13]] -> indices 4,5,6,7
    # A+ = A0 + A1
    # A- = A0 - A1
    
    # Indices for B (4x4):
    # B00 = [[B00, B01], [B02, B03]] -> indices 0,1,2,3
    # B01 = [[B04, B05], [B06, B07]] -> indices 4,5,6,7
    # B10 = [[B08, B09], [B10, B11]] -> indices 8,9,10,11
    # B11 = [[B12, B13], [B14, B15]] -> indices 12,13,14,15
    
    # B00+B10:
    # [[B00+B08, B01+B09], [B02+B10, B03+B11]]
    # B00-B10:
    # [[B00-B08, B01-B09], [B02-B10, B03-B11]]
    # B01+B11:
    # [[B04+B12, B05+B13], [B06+B14, B07+B15]]
    # B01-B11:
    # [[B04-B12, B05-B13], [B06-B14, B07-B15]]
    
    # We define 4 sets of 7 Strassen intermediates.
    # Set 1: A+ * (B00+B10)
    # Set 2: A- * (B00-B10)
    # Set 3: A+ * (B01+B11)
    # Set 4: A- * (B01-B11)
    
    # Helper to add Strassen terms
    # x_idx: indices for X in U (4 indices for 2x2)
    # y_idx: indices for Y in V (4 indices for 2x2)
    # w_idx: indices for Z in W (4 indices for 2x2)
    # sign: multiplier for the result (1 or -1)
    # col_offset: starting column in U, V, W
    
    def add_strassen(x_idx, y_idx, w_idx, sign, col_offset):
        # x_idx: [x00, x01, x10, x11]
        # y_idx: [y00, y01, y10, y11]
        # w_idx: [z00, z01, z10, z11]
        
        # M1 = (x00+x11)(y00+y11)
        U[x_idx[0]][col_offset] += sign
        U[x_idx[3]][col_offset] += sign
        V[y_idx[0]][col_offset] += sign
        V[y_idx[3]][col_offset] += sign
        W[w_idx[0]][col_offset] += sign
        W[w_idx[3]][col_offset] += -sign
        # Note: M1 contributes to z00 (+1) and z11 (-1)
        # Wait, z00 = M1 - M3 + M4 + M5. So M1 coeff is +1.
        # z11 = M1 - M2 + M6 + M7. So M1 coeff is +1.
        # My W update above was wrong.
        # Let's do it term by term.
        
        # Reset W for this column
        W[w_idx[0]][col_offset] = 0
        W[w_idx[1]][col_offset] = 0
        W[w_idx[2]][col_offset] = 0
        W[w_idx[3]][col_offset] = 0
        
        # M1
        U[x_idx[0]][col_offset] += sign
        U[x_idx[3]][col_offset] += sign
        V[y_idx[0]][col_offset] += sign
        V[y_idx[3]][col_offset] += sign
        W[w_idx[0]][col_offset] += sign
        W[w_idx[3]][col_offset] += sign
        
        # M2 = (x10+x11)y00
        col_offset += 1
        U[x_idx[2]][col_offset] += sign
        U[x_idx[3]][col_offset] += sign
        V[y_idx[0]][col_offset] += sign
        W[w_idx[2]][col_offset] += sign
        W[w_idx[3]][col_offset] += -sign
        
        # M3 = x00(y01-y11)
        col_offset += 1
        U[x_idx[0]][col_offset] += sign
        V[y_idx[1]][col_offset] += sign
        V[y_idx[3]][col_offset] += -sign
        W[w_idx[0]][col_offset] += -sign
        W[w_idx[1]][col_offset] += sign
        
        # M4 = x11(y10-y00)
        col_offset += 1
        U[x_idx[3]][col_offset] += sign
        V[y_idx[2]][col_offset] += sign
        V[y_idx[0]][col_offset] += -sign
        W[w_idx[0]][col_offset] += sign
        W[w_idx[2]][col_offset] += sign
        
        # M5 = (x00+x01)y11
        col_offset += 1
        U[x_idx[0]][col_offset] += sign
        U[x_idx[1]][col_offset] += sign
        V[y_idx[3]][col_offset] += sign
        W[w_idx[0]][col_offset] += sign
        W[w_idx[1]][col_offset] += sign
        
        # M6 = (x10-x00)(y00+y01)
        col_offset += 1
        U[x_idx[2]][col_offset] += sign
        U[x_idx[0]][col_offset] += -sign
        V[y_idx[0]][col_offset] += sign
        V[y_idx[1]][col_offset] += sign
        W[w_idx[3]][col_offset] += sign
        
        # M7 = (x01-x11)(y10+y11)
        col_offset += 1
        U[x_idx[1]][col_offset] += sign
        U[x_idx[3]][col_offset] += -sign
        V[y_idx[2]][col_offset] += sign
        V[y_idx[3]][col_offset] += sign
        W[w_idx[1]][col_offset] += -sign
        W[w_idx[2]][col_offset] += sign
        W[w_idx[3]][col_offset] += sign
        
        return col_offset + 1

    # A+ indices:
    # A+00 = A00 + A10 (U[0] + U[4])
    # A+01 = A01 + A11 (U[1] + U[5])
    # A+10 = A02 + A12 (U[2] + U[6])
    # A+11 = A03 + A13 (U[3] + U[7])
    # This is not a simple index list. We need to add contributions to U.
    
    # Let's redefine the approach.
    # We will fill U, V, W directly.
    
    # A+ = A0 + A1
    # A- = A0 - A1
    
    # B00+B10 indices in V:
    # (B00+B10)00 = B00 + B08 (V[0] + V[8])
    # (B00+B10)01 = B01 + B09 (V[1] + V[9])
    # (B00+B10)10 = B02 + B10 (V[2] + V[10])
    # (B00+B10)11 = B03 + B11 (V[3] + V[11])
    
    # We need a helper that takes linear combinations.
    
    def add_strassen_lin(x_lin, y_lin, w_lin, sign, col_offset):
        # x_lin: list of (index, coeff) for U
        # y_lin: list of (index, coeff) for V
        # w_lin: list of (index, coeff) for W
        
        # M1 = (x00+x11)(y00+y11)
        # x00 = sum x_lin[0], x11 = sum x_lin[3]
        # y00 = sum y_lin[0], y11 = sum y_lin[3]
        
        # Term: (x00+x11) * (y00+y11)
        # This is a product of two linear forms.
        # It expands to 4 terms.
        # But we are adding to the tensor decomposition.
        # Each bilinear form is a product of a linear form in A and a linear form in B.
        # So we just need to add the coefficients to U and V.
        
        # For M1:
        # U coeff for x00: sign * 1
        # U coeff for x11: sign * 1
        # V coeff for y00: sign * 1
        # V coeff for y11: sign * 1
        # W coeff for z00: sign * 1
        # W coeff for z11: sign * 1
        
        # We need to map x_lin[0] to the actual U indices.
        
        # Let's assume x_lin is a list of 4 items, each a list of (idx, coeff).
        
        for i in range(4):
            for idx, coeff in x_lin[i]:
                U[idx][col_offset] += sign * coeff
                # Wait, M1 uses x00 and x11.
                # So we only add to U for x00 and x11.
                
        # This is getting messy. Let's just hardcode the 28 forms.
        
    # Hardcoding is safer.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A+ = A0 + A1
    # A- = A0 - A1
    # A+00: U[0] + U[4]
    # A+01: U[1] + U[5]
    # A+10: U[2] + U[6]
    # A+11: U[3] + U[7]
    
    # A-00: U[0] - U[4]
    # A-01: U[1] - U[5]
    # A-10: U[2] - U[6]
    # A-11: U[3] - U[7]
    
    # B00+B10:
    # V0: V[0] + V[8]
    # V1: V[1] + V[9]
    # V2: V[2] + V[10]
    # V3: V[3] + V[11]
    
    # B00-B10:
    # V0: V[0] - V[8]
    # V1: V[1] - V[9]
    # V2: V[2] - V[10]
    # V3: V[3] - V[11]
    
    # B01+B11:
    # V0: V[4] + V[12]
    # V1: V[5] + V[13]
    # V2: V[6] + V[14]
    # V3: V[7] + V[15]
    
    # B01-B11:
    # V0: V[4] - V[12]
    # V1: V[5] - V[13]
    # V2: V[6] - V[14]
    # V3: V[7] - V[15]
    
    # C0 = 1/2 (P1 - P2)
    # C1 = 1/2 (P3 - P4)
    
    # P1 = A+ * (B00+B10)
    # P2 = A- * (B00-B10)
    # P3 = A+ * (B01+B11)
    # P4 = A- * (B01-B11)
    
    # We will add 7 forms for P1, 7 for P2, 7 for P3, 7 for P4.
    # Total 28.
    
    # Strassen for P1 (A+ * B00+B10)
    # x = A+, y = B00+B10
    # M1: (x00+x11)(y00+y11)
    # U: A+00, A+11 -> U[0], U[4], U[3], U[7]
    # V: y00, y11 -> V[0], V[8], V[3], V[11]
    # W: z00, z11 -> W[0], W[3] (C00, C01, C10, C11 of C0)
    # C0 indices in W: 0,1,2,3
    
    col = 0
    # P1
    # M1
    U[0][col] += 1; U[4][col] += 1; U[3][col] += 1; U[7][col] += 1
    V[0][col] += 1; V[8][col] += 1; V[3][col] += 1; V[11][col] += 1
    W[0][col] += 1; W[3][col] += 1
    col += 1
    
    # M2: (x10+x11)y00
    U[2][col] += 1; U[6][col] += 1; U[3][col] += 1; U[7][col] += 1
    V[0][col] += 1; V[8][col] += 1
    W[2][col] += 1; W[3][col] += -1
    col += 1
    
    # M3: x00(y01-y11)
    U[0][col] += 1; U[4][col] += 1
    V[1][col] += 1; V[9][col] += 1; V[3][col] += -1; V[11][col] += -1
    W[0][col] += -1; W[1][col] += 1
    col += 1
    
    # M4: x11(y10-y00)
    U[3][col] += 1; U[7][col] += 1
    V[2][col] += 1; V[10][col] += 1; V[0][col] += -1; V[8][col] += -1
    W[0][col] += 1; W[2][col] += 1
    col += 1
    
    # M5: (x00+x01)y11
    U[0][col] += 1; U[4][col] += 1; U[1][col] += 1; U[5][col] += 1
    V[3][col] += 1; V[11][col] += 1
    W[0][col] += 1; W[1][col] += 1
    col += 1
    
    # M6: (x10-x00)(y00+y01)
    U[2][col] += 1; U[6][col] += 1; U[0][col] += -1; U[4][col] += -1
    V[0][col] += 1; V[8][col] += 1; V[1][col] += 1; V[9][col] += 1
    W[3][col] += 1
    col += 1
    
    # M7: (x01-x11)(y10+y11)
    U[1][col] += 1; U[5][col] += 1; U[3][col] += -1; U[7][col] += -1
    V[2][col] += 1; V[10][col] += 1; V[3][col] += 1; V[11][col] += 1
    W[1][col] += -1; W[2][col] += 1; W[3][col] += 1
    col += 1
    
    # P2 (A- * B00-B10) with sign -1
    # x = A-, y = B00-B10
    # M1
    U[0][col] += -1; U[4][col] += 1; U[3][col] += -1; U[7][col] += 1
    V[0][col] += 1; V[8][col] += -1; V[3][col] += 1; V[11][col] += -1
    W[0][col] += -1; W[3][col] += -1
    col += 1
    
    # M2
    U[2][col] += -1; U[6][col] += 1; U[3][col] += -1; U[7][col] += 1
    V[0][col] += 1; V[8][col] += -1
    W[2][col] += -1; W[3][col] += 1
    col += 1
    
    # M3
    U[0][col] += -1; U[4][col] += 1
    V[1][col] += 1; V[9][col] += -1; V[3][col] += -1; V[11][col] += 1
    W[0][col] += 1; W[1][col] += -1
    col += 1
    
    # M4
    U[3][col] += -1; U[7][col] += 1
    V[2][col] += 1; V[10][col] += -1; V[0][col] += -1; V[8][col] += 1
    W[0][col] += -1; W[2][col] += -1
    col += 1
    
    # M5
    U[0][col] += -1; U[4][col] += 1; U[1][col] += -1; U[5][col] += 1
    V[3][col] += 1; V[11][col] += -1
    W[0][col] += -1; W[1][col] += -1
    col += 1
    
    # M6
    U[2][col] += -1; U[6][col] += 1; U[0][col] += 1; U[4][col] += -1
    V[0][col] += 1; V[8][col] += -1; V[1][col] += 1; V[9][col] += -1
    W[3][col] += -1
    col += 1
    
    # M7
    U[1][col] += -1; U[5][col] += 1; U[3][col] += 1; U[7][col] += -1
    V[2][col] += 1; V[10][col] += -1; V[3][col] += 1; V[11][col] += -1
    W[1][col] += 1; W[2][col] += -1; W[3][col] += -1
    col += 1
    
    # P3 (A+ * B01+B11)
    # x = A+, y = B01+B11
    # y indices: 4,5,6,7 and 12,13,14,15
    # M1
    U[0][col] += 1; U[4][col] += 1; U[3][col] += 1; U[7][col] += 1
    V[4][col] += 1; V[12][col] += 1; V[7][col] += 1; V[15][col] += 1
    W[4][col] += 1; W[7][col] += 1
    col += 1
    
    # M2
    U[2][col] += 1; U[6][col] += 1; U[3][col] += 1; U[7][col] += 1
    V[4][col] += 1; V[12][col] += 1
    W[6][col] += 1; W[7][col] += -1
    col += 1
    
    # M3
    U[0][col] += 1; U[4][col] += 1
    V[5][col] += 1; V[13][col] += 1; V[7][col] += -1; V[15][col] += -1
    W[4][col] += -1; W[5][col] += 1
    col += 1
    
    # M4
    U[3][col] += 1; U[7][col] += 1
    V[6][col] += 1; V[14][col] += 1; V[4][col] += -1; V[12][col] += -1
    W[4][col] += 1; W[6][col] += 1
    col += 1
    
    # M5
    U[0][col] += 1; U[4][col] += 1; U[1][col] += 1; U[5][col] += 1
    V[7][col] += 1; V[15][col] += 1
    W[4][col] += 1; W[5][col] += 1
    col += 1
    
    # M6
    U[2][col] += 1; U[6][col] += 1; U[0][col] += -1; U[4][col] += -1
    V[4][col] += 1; V[12][col] += 1; V[5][col] += 1; V[13][col] += 1
    W[7][col] += 1
    col += 1
    
    # M7
    U[1][col] += 1; U[5][col] += 1; U[3][col] += -1; U[7][col] += -1
    V[6][col] += 1; V[14][col] += 1; V[7][col] += 1; V[15][col] += 1
    W[5][col] += -1; W[6][col] += 1; W[7][col] += 1
    col += 1
    
    # P4 (A- * B01-B11) with sign -1
    # x = A-, y = B01-B11
    # M1
    U[0][col] += -1; U[4][col] += 1; U[3][col] += -1; U[7][col] += 1
    V[4][col] += 1; V[12][col] += -1; V[7][col] += 1; V[15][col] += -1
    W[4][col] += -1; W[7][col] += -1
    col += 1
    
    # M2
    U[2][col] += -1; U[6][col] += 1; U[3][col] += -1; U[7][col] += 1
    V[4][col] += 1; V[12][col] += -1
    W[6][col] += -1; W[7][col] += 1
    col += 1
    
    # M3
    U[0][col] += -1; U[4][col] += 1
    V[5][col] += 1; V[13][col] += -1; V[7][col] += -1; V[15][col] += 1
    W[4][col] += 1; W[5][col] += -1
    col += 1
    
    # M4
    U[3][col] += -1; U[7][col] += 1
    V[6][col] += 1; V[14][col] += -1; V[4][col] += -1; V[12][col] += 1
    W[4][col] += -1; W[6][col] += -1
    col += 1
    
    # M5
    U[0][col] += -1; U[4][col] += 1; U[1][col] += -1; U[5][col] += 1
    V[7][col] += 1; V[15][col] += -1
    W[4][col] += -1; W[5][col] += -1
    col += 1
    
    # M6
    U[2][col] += -1; U[6][col] += 1; U[0][col] += 1; U[4][col] += -1
    V[4][col] += 1; V[12][col] += -1; V[5][col] += 1; V[13][col] += -1
    W[7][col] += -1
    col += 1
    
    # M7
    U[1][col] += -1; U[5][col] += 1; U[3][col] += 1; U[7][col] += -1
    V[6][col] += 1; V[14][col] += -1; V[7][col] += 1; V[15][col] += -1
    W[5][col] += 1; W[6][col] += -1; W[7][col] += -1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
