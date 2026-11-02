def laderman_3x3x3():
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Indices: i, j, k in 0..2
    # A index: i*3+j
    # B index: j*3+k
    # C index: k*3+i
    
    # Term 0: A00 * B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A01 * B10
    U[1, 1] = 1; V[3, 1] = 1; W[1, 1] = 1 # B10 is j=1, k=0 -> idx 3
    # Term 2: A02 * B20
    U[2, 2] = 1; V[6, 2] = 1; W[2, 2] = 1 # B20 is j=2, k=0 -> idx 6
    # Term 3: A10 * B01
    U[3, 3] = 1; V[1, 3] = 1; W[0, 3] = 1 # B01 is j=0, k=1 -> idx 1
    # Term 4: A11 * B11
    U[4, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
    # Term 5: A12 * B21
    U[5, 5] = 1; V[7, 5] = 1; W[2, 5] = 1 # B21 is j=2, k=1 -> idx 7
    # Term 6: A20 * B02
    U[6, 6] = 1; V[2, 6] = 1; W[0, 6] = 1 # B02 is j=0, k=2 -> idx 2
    # Term 7: A21 * B12
    U[7, 7] = 1; V[5, 7] = 1; W[1, 7] = 1 # B12 is j=1, k=2 -> idx 5
    # Term 8: A22 * B22
    U[8, 8] = 1; V[8, 8] = 1; W[2, 8] = 1
    
    # Term 9: (A00 + A11) * (B11 + B00)
    U[0, 9] = 1; U[4, 9] = 1
    V[4, 9] = 1; V[0, 9] = 1
    W[1, 9] = 1; W[0, 9] = 1
    
    # Term 10: (A01 + A12) * (B22 + B00)? No.
    # Laderman uses specific combinations.
    # I will use a verified set from a known source or code.
    # Since I cannot verify offline, I will use a fallback:
    # If Laderman validation fails, use schoolbook.
    
    # To be safe, I will just use schoolbook for the 3x3x3 block too?
    # Rank 9 for 3x3x3.
    # Total rank 9+9+9+9 = 36? No.
    # 3x3x8 = 3x3x4 + 3x3x4.
    # 3x3x4 = 3x3x3 + 3x3x1.
    # If I use schoolbook for 3x3x3 (rank 9) and 3x3x1 (rank 9).
    # Total 9+9+9+9 = 36.
    # This is much better than 72!
    # Why did I think Laderman was needed?
    # Schoolbook for 3x3x3 is rank 9.
    # Schoolbook for 3x3x1 is rank 9.
    # So 3x3x4 is rank 18.
    # 3x3x8 is rank 36.
    
    # Let's verify this.
    # C = A * B.
    # Split B into B1 (3x4) and B2 (3x4).
    # C1 = A * B1.
    # Split B1 into B11 (3x3) and B12 (3x1).
    # C11 = A * B11 (3x3). Rank 9.
    # C12 = A * B12 (3x1). Rank 9.
    # Total rank for C1 is 18.
    # Same for C2.
    # Total rank 36.
    
    # This is a massive improvement over 72.
    # Why did the seed use 72?
    # Because it used schoolbook for the WHOLE 3x3x8.
    # 3*3*8 = 72.
    # By splitting, we reduce the rank?
    # Yes, because the rank of a direct sum is the sum of ranks.
    # And the rank of 3x3x4 is NOT 3*3*4 = 36?
    # Wait.
    # Rank of 3x3x3 is 9 (schoolbook).
    # Rank of 3x3x1 is 9 (schoolbook).
    # So Rank(3x3x4) <= 18.
    # But is Rank(3x3x4) actually 18?
    # The tensor for 3x3x4 is NOT the direct sum of 3x3x3 and 3x3x1?
    # It IS, because the output columns are independent.
    # C1 (cols 0-2) depends only on B1 (cols 0-2).
    # C2 (col 3) depends only on B2 (col 3).
    # So yes, it is a direct sum.
    
    # So I can achieve rank 36 easily!
    
    return U, V, W
