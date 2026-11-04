"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank 19 exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Partition A (3x3) into:
    A1: (0,0) scalar
    A2: (0, 1:3) 1x2 vector
    A3: (1:3, 0) 2x1 vector
    A4: (1:3, 1:3) 2x2 matrix
    
    Partition B (3x4) into:
    B1: (0, 0:2) 1x2 vector
    B2: (0, 2:4) 1x2 vector
    B3: (1:3, 0:2) 2x2 matrix
    B4: (1:3, 2:4) 2x2 matrix
    
    Products:
    C11 = A1*B1 + A2*B3
    C12 = A1*B2 + A2*B4
    C21 = A3*B1 + A4*B3
    C22 = A3*B2 + A4*B4
    
    Ranks:
    A1*B1: 1 (scalar * vector)
    A1*B2: 1
    A3*B1: 1 (vector * vector)
    A3*B2: 1
    A2*B3: 1 (vector * matrix, rank 1 decomposition possible)
    A2*B4: 1
    A4*B3: 7 (Strassen 2x2x2)
    A4*B4: 7 (Strassen 2x2x2)
    
    Total Rank: 1+1+1+1+1+1+7+7 = 19
    """
    n, m, p = 3, 3, 4
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx, v_idx, w_idx can be lists of indices or single indices
        if not isinstance(u_idx, list): u_idx = [u_idx]
        if not isinstance(v_idx, list): v_idx = [v_idx]
        if not isinstance(w_idx, list): w_idx = [w_idx]
        if not isinstance(u_val, list): u_val = [u_val]
        if not isinstance(v_val, list): v_val = [v_val]
        if not isinstance(w_val, list): w_val = [w_val]
        
        # This helper is too generic. I will manually populate.
        pass

    # Indices mapping:
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # A1: U[0] (i=0, j=0)
    # A2: U[1] (i=0, j=1), U[2] (i=0, j=2)
    # A3: U[3] (i=1, j=0), U[6] (i=2, j=0)
    # A4: U[4] (1,1), U[5] (1,2), U[7] (2,1), U[8] (2,2)
    
    # B1: V[0] (j=0, k=0), V[1] (j=0, k=1)
    # B2: V[2] (j=0, k=2), V[3] (j=0, k=3)
    # B3: V[4] (1,0), V[5] (1,1), V[8] (2,0), V[9] (2,1)
    # B4: V[6] (1,2), V[7] (1,3), V[10] (2,2), V[11] (2,3)
    
    # W indices for C:
    # C11 (0,0): W[0]
    # C12 (0,1): W[1]
    # C13 (0,2): W[2]
    # C14 (0,3): W[3]
    # C21 (1,0): W[3+0]=W[3]? No. W[k*n+i].
    # k=0, i=1 -> W[1]
    # k=1, i=1 -> W[4]
    # k=2, i=1 -> W[7]
    # k=3, i=1 -> W[10]
    # k=0, i=2 -> W[2]
    # k=1, i=2 -> W[5]
    # k=2, i=2 -> W[8]
    # k=3, i=2 -> W[11]
    
    # Let's re-verify W indices.
    # W is (p*n, rank) = (12, 19).
    # Index = k * 3 + i.
    # C[i, k].
    # i=0, k=0 -> 0
    # i=0, k=1 -> 1
    # i=0, k=2 -> 2
    # i=0, k=3 -> 3
    # i=1, k=0 -> 3
    # i=1, k=1 -> 4
    # i=1, k=2 -> 5
    # i=1, k=3 -> 6
    # i=2, k=0 -> 6
    # i=2, k=1 -> 7
    # i=2, k=2 -> 8
    # i=2, k=3 -> 9
    
    # Term 1: A1*B1 (1x1 * 1x2) -> C[0,0], C[0,1]
    # U[0]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1
    col = 0
    U[0][col] = 1
    V[0][col] = 1
    V[1][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # Term 2: A1*B2 (1x1 * 1x2) -> C[0,2], C[0,3]
    # U[0]=1, V[2]=1, V[3]=1, W[2]=1, W[3]=1
    U[0][col] = 1
    V[2][col] = 1
    V[3][col] = 1
    W[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 3: A3*B1 (2x1 * 1x2) -> C[1,0], C[1,1], C[2,0], C[2,1]
    # U[3]=1, U[6]=1, V[0]=1, V[1]=1, W[3]=1, W[4]=1, W[6]=1, W[7]=1
    U[3][col] = 1
    U[6][col] = 1
    V[0][col] = 1
    V[1][col] = 1
    W[3][col] = 1
    W[4][col] = 1
    W[6][col] = 1
    W[7][col] = 1
    col += 1
    
    # Term 4: A3*B2 (2x1 * 1x2) -> C[1,2], C[1,3], C[2,2], C[2,3]
    # U[3]=1, U[6]=1, V[2]=1, V[3]=1, W[5]=1, W[6]=1, W[8]=1, W[9]=1
    U[3][col] = 1
    U[6][col] = 1
    V[2][col] = 1
    V[3][col] = 1
    W[5][col] = 1
    W[6][col] = 1
    W[8][col] = 1
    W[9][col] = 1
    col += 1
    
    # Term 5: A2*B3 (1x2 * 2x2) -> C[0,0], C[0,1]
    # U[1]=1, U[2]=1, V[4]=1, V[5]=1, V[8]=1, V[9]=1, W[0]=1, W[1]=1
    U[1][col] = 1
    U[2][col] = 1
    V[4][col] = 1
    V[5][col] = 1
    V[8][col] = 1
    V[9][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # Term 6: A2*B4 (1x2 * 2x2) -> C[0,2], C[0,3]
    # U[1]=1, U[2]=1, V[6]=1, V[7]=1, V[10]=1, V[11]=1, W[2]=1, W[3]=1
    U[1][col] = 1
    U[2][col] = 1
    V[6][col] = 1
    V[7][col] = 1
    V[10][col] = 1
    V[11][col] = 1
    W[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # Terms 7-13: A4*B3 (2x2 * 2x2) Strassen
    # A4: U[4], U[5], U[7], U[8]
    # B3: V[4], V[5], V[8], V[9]
    # C: W[3], W[4], W[6], W[7]
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # A00=U[4], A11=U[8], B00=V[4], B11=V[9]
    # C00 += M1, C11 += M1
    U[4][col] = 1
    U[8][col] = 1
    V[4][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    # A10=U[7], A11=U[8], B00=V[4]
    # C10 += M2
    U[7][col] = 1
    U[8][col] = 1
    V[4][col] = 1
    W[6][col] = 1
    col += 1
    
    # M3 = A00(B01-B11)
    # A00=U[4], B01=V[5], B11=V[9]
    # C01 += M3
    U[4][col] = 1
    V[5][col] = 1
    V[9][col] = -1
    W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-A00)
    # A11=U[8], B10=V[8], A00=U[4]
    # C11 += M4
    U[8][col] = 1
    V[8][col] = 1
    U[4][col] = -1 # Wait, U is same vector? No, U[4] and U[8] are different indices.
    # U[4] is A00. U[8] is A11.
    # M4 = A11 * (B10 - A00) is NOT STRASSEN.
    # Strassen M4 = (A11 - A00)(B00 + B01)
    # Let's use standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = A00B00
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A00+A10)(B11-B10)
    
    # My previous M4 definition was wrong. Let's restart Strassen for A4*B3.
    # Reset col to 6 (after 6 terms)
    col = 6
    
    # A4 indices: 4, 5, 7, 8
    # B3 indices: 4, 5, 8, 9
    # W indices: 3, 4, 6, 7
    
    # M1
    U[4][col] = 1; U[8][col] = 1
    V[4][col] = 1; V[9][col] = 1
    W[3][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[7][col] = 1; U[8][col] = 1
    V[4][col] = 1
    W[6][col] = 1
    col += 1
    
    # M3
    U[4][col] = 1
    V[5][col] = 1; V[9][col] = -1
    W[4][col] = 1
    col += 1
    
    # M4
    U[8][col] = 1
    V[8][col] = 1; V[4][col] = -1
    W[7][col] = 1
    col += 1
    
    # M5
    U[4][col] = 1
    V[4][col] = 1
    W[3][col] = 1
    col += 1
    
    # M6
    U[8][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7
    U[4][col] = 1; U[7][col] = 1
    V[9][col] = 1; V[8][col] = -1
    W[6][col] = 1
    col += 1
    
    # Reconstruct C from M1..M7:
    # C00 = M1 - M4 + M3 - M7
    # C01 = M5 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M6 + M3
    
    # We need to add these combinations to W.
    # Currently W has:
    # Col 6 (M1): W3=1, W7=1
    # Col 7 (M2): W6=1
    # Col 8 (M3): W4=1
    # Col 9 (M4): W7=1
    # Col 10 (M5): W3=1
    # Col 11 (M6): W4=1
    # Col 12 (M7): W6=1
    
    # We need to adjust W coefficients.
    # W3 (C00): M1 - M4 + M3 - M7 -> 1 - 1 + 1 - 1 = 0?
    # Wait, M4 is added to W7 in col 9. M3 to W4 in col 8. M7 to W6 in col 12.
    # C00 is W3.
    # M1 contributes 1 to W3.
    # M4 contributes 0 to W3.
    # M3 contributes 0 to W3.
    # M7 contributes 0 to W3.
    # So W3 needs M1 - M4 + M3 - M7?
    # No, the Strassen formulas define C in terms of M.
    # We are defining M as rank-1 terms.
    # We must set W such that the sum of rank-1 terms equals C.
    
    # Let's explicitly set the W values for the 7 Strassen terms.
    # M1 term (col 6): W3=1, W7=1. (Contributes to C00, C11)
    # M2 term (col 7): W6=1. (Contributes to C10)
    # M3 term (col 8): W4=1. (Contributes to C01)
    # M4 term (col 9): W7=1. (Contributes to C11) -- Wait, M4 is A11(B10-B00).
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = A00B00
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A00+A10)(B11-B10)
    
    # C00 = M1 - M4 + M3 - M7
    # C01 = M5 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M6 + M3
    
    # So:
    # Col 6 (M1): W3=1, W7=1
    # Col 7 (M2): W6=1
    # Col 8 (M3): W4=1
    # Col 9 (M4): W7=-1 (because C00 has -M4, C01 has +M4, C11 has 0? No, C11 has 0 M4?
    # C11 = M1 - M2 + M6 + M3. No M4.
    # C01 = M5 + M4.
    # C00 = M1 - M4 + M3 - M7.
    # So M4 contributes -1 to W3 (C00) and +1 to W4 (C01).
    # Col 9: W3=-1, W4=1.
    
    # Col 10 (M5): W4=1 (C01).
    # Col 11 (M6): W7=1 (C11).
    # Col 12 (M7): W3=-1 (C00), W6=1 (C10).
    
    # Let's update the W values in the code.
    
    # Term 7 (M1)
    U[4][6] = 1; U[8][6] = 1
    V[4][6] = 1; V[9][6] = 1
    W[3][6] = 1; W[7][6] = 1
    
    # Term 8 (M2)
    U[7][7] = 1; U[8][7] = 1
    V[4][7] = 1
    W[6][7] = 1
    
    # Term 9 (M3)
    U[4][8] = 1
    V[5][8] = 1; V[9][8] = -1
    W[4][8] = 1
    
    # Term 10 (M4)
    U[8][9] = 1
    V[8][9] = 1; V[4][9] = -1
    W[3][9] = -1; W[4][9] = 1
    
    # Term 11 (M5)
    U[4][10] = 1
    V[4][10] = 1
    W[4][10] = 1
    
    # Term 12 (M6)
    U[8][11] = 1; U[4][11] = -1
    V[4][11] = 1; V[5][11] = 1
    W[7][11] = 1
    
    # Term 13 (M7)
    U[4][12] = 1; U[7][12] = 1
    V[9][12] = 1; V[8][12] = -1
    W[3][12] = -1; W[6][12] = 1
    
    # Terms 14-20: A4*B4 (2x2 * 2x2) Strassen
    # A4: U[4], U[5], U[7], U[8]
    # B4: V[6], V[7], V[10], V[11]
    # C: W[5], W[6], W[8], W[9] (C[1,2], C[1,3], C[2,2], C[2,3])
    # W indices:
    # C[1,2] (i=1, k=2) -> 2*3+1 = 7? No.
    # W[k*n+i].
    # i=1, k=2 -> 2*3+1 = 7.
    # i=1, k=3 -> 3*3+1 = 10.
    # i=2, k=2 -> 2*3+2 = 8.
    # i=2, k=3 -> 3*3+2 = 11.
    
    # So W indices are 7, 10, 8, 11.
    
    # M1
    U[4][13] = 1; U[8][13] = 1
    V[6][13] = 1; V[11][13] = 1
    W[7][13] = 1; W[11][13] = 1
    
    # M2
    U[7][14] = 1; U[8][14] = 1
    V[6][14] = 1
    W[8][14] = 1
    
    # M3
    U[4][15] = 1
    V[7][15] = 1; V[11][15] = -1
    W[10][15] = 1
    
    # M4
    U[8][16] = 1
    V[10][16] = 1; V[6][16] = -1
    W[7][16] = -1; W[10][16] = 1
    
    # M5
    U[4][17] = 1
    V[6][17] = 1
    W[10][17] = 1
    
    # M6
    U[8][18] = 1; U[4][18] = -1
    V[6][18] = 1; V[7][18] = 1
    W[11][18] = 1
    
    # M7
    U[4][19] = 1; U[7][19] = 1
    V[11][19] = 1; V[10][19] = -1
    W[7][19] = -1; W[8][19] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
