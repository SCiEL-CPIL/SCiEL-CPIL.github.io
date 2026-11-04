import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2 (Rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1
    
    # 2. Laderman decomposition for 3x3 (Rank 7)
    # Laderman's algorithm for 3x3 matrix multiplication with rank 7.
    # U3: (9, 7), V3: (9, 7), W3: (9, 7)
    # Indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    
    U3 = np.zeros((9, 7), dtype=float)
    V3 = np.zeros((9, 7), dtype=float)
    W3 = np.zeros((9, 7), dtype=float)
    
    # Laderman's 7 products for 3x3:
    # Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    # Let C = A*B
    #
    # M1 = (a11+a21+a31)(b11+b21+b31)
    # M2 = (a11+a21-a31)(b11-b21+b31) 
    # M3 = (a11-a21+a31)(b11+b21-b31)
    # M4 = (a11-a21-a31)(b11-b21-b31)
    # M5 = (a21+a31)(b21+b31)
    # M6 = (a21-a31)(b21-b31)
    # M7 = a31*b31
    #
    # Wait, that's not right. Let me look up the actual Laderman algorithm.
    # Laderman 1976: 3x3 with 7 multiplications.
    #
    # Actually, the standard Laderman algorithm is:
    # M1 = (a11+a21)(b11+b21)
    # M2 = (a11-a21)(b11-b21)
    # M3 = (a11+a21+a31)(b11+b21+b31)
    # M4 = (a11+a21-a31)(b11+b21-b31)
    # M5 = (a11-a21+a31)(b11-b21+b31)
    # M6 = (a21+a31)(b21+b31)
    # M7 = (a21-a31)(b21-b31)
    #
    # Hmm, I need to be more careful. Let me use a known correct Laderman decomposition.
    #
    # From literature, Laderman's 3x3 rank-7 algorithm:
    # The 7 bilinear forms are:
    # P1 = (a11+a21+a31)(b11+b21+b31)
    # P2 = (a11+a21-a31)(b11+b21-b31)
    # P3 = (a11-a21+a31)(b11-b21+b31)
    # P4 = (a11-a21-a31)(b11-b21-b31)
    # P5 = (a21+a31)(b21+b31)
    # P6 = (a21-a31)(b21-b31)
    # P7 = a31*b31
    #
    # This doesn't seem right either. Let me use a different approach.
    #
    # Actually, I recall that Laderman's algorithm for 3x3 uses:
    # M1 = (a11 + a21 + a31)(b11 + b21 + b31)
    # M2 = (a11 + a21 - a31)(b11 + b21 - b31)
    # M3 = (a11 - a21 + a31)(b11 - b21 + b31)
    # M4 = (a11 - a21 - a31)(b11 - b21 - b31)
    # M5 = (a21 + a31)(b21 + b31)
    # M6 = (a21 - a31)(b21 - b31)
    # M7 = a31 * b31
    #
    # But this only covers the first column of C. We need all of C.
    #
    # Let me look at this differently. The Laderman algorithm actually computes
    # all 9 entries of C using 7 multiplications. Let me use the correct formulas.
    #
    # From Laderman (1976), "A non-linear algorithm for matrix multiplication":
    # For 3x3 matrices, the following 7 products suffice:
    #
    # Let's use a verified version. I'll construct it by ensuring correctness.
    #
    # Actually, let me use a simpler approach: I'll verify by construction.
    # The key insight is that we need U3, V3, W3 such that:
    # sum_r U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r] = C[i,k]
    # for all i,k in {0,1,2} and all entries of A and B.
    #
    # Let me use the following known Laderman decomposition:
    #
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a00-a10)(b00-b10)  
    # M3 = (a00+a10+a20)(b00+b10+b20)
    # M4 = (a00+a10-a20)(b00+b10-b20)
    # M5 = (a00-a10+a20)(b00-b10+b20)
    # M6 = (a10+a20)(b10+b20)
    # M7 = (a10-a20)(b10-b20)
    #
    # This still only gives column 0. I think I'm confusing things.
    #
    # Let me just use the trivial 3x3 (rank 27) and accept rank 7*27=189,
    # OR let me find a proper Laderman implementation.
    #
    # After more thought, let me use a verified Laderman 3x3 rank-7 decomposition.
    # The correct Laderman algorithm for 3x3:
    #
    # The 7 multiplications are:
    # M1 = (a00+a10+a20)(b00+b10+b20)
    # M2 = (a00+a10-a20)(b00+b10-b20)
    # M3 = (a00-a10+a20)(b00-b10+b20)
    # M4 = (a00-a10-a20)(b00-b10-b20)
    # M5 = (a10+a20)(b10+b20)
    # M6 = (a10-a20)(b10-b20)
    # M7 = a20*b20
    #
    # This gives C[0,0] = (M1+M2+M3+M4)/4 + ... no, this is getting complicated.
    #
    # Let me use a different strategy: I'll implement the Kronecker product
    # using Strassen 2x2 (rank 7) and Trivial 3x3 (rank 27), giving rank 189.
    # But 189 > 125, so that's worse.
    #
    # Alternative: Use Strassen 2x2 (rank 7) and Strassen 2x2 (rank 7) to get
    # 4x4 (rank 49), then pad to 5x5? Can't extract 5x5 from 4x4.
    #
    # Best option: Just use trivial 5x5 (rank 125) which is already in the current code.
    # But the directive asks for rank 49.
    #
    # Let me try: 5x5 can be seen as a sub-matrix of 6x6.
    # 6x6 = 2x2 ⊗ 3x3.
    # If I have rank-7 for 2x2 and rank-7 for 3x3, I get rank 49 for 6x6.
    # Then I extract 5x5 by dropping one row and one column.
    #
    # So I NEED a valid rank-7 3x3 decomposition. Let me find one.
    #
    # After research, here's a valid Laderman 3x3 rank-7 decomposition:
    #
    # U3 (9x7):
    # Row indices: i*3+j for i,j in 0,1,2
    #
    # Let me define it explicitly and verify.
    #
    # I'll use the following construction based on Laderman's paper:
    #
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a00-a10)(b00-b10)
    # M3 = (a00+a10+a20)(b00+b10+b20)
    # M4 = (a00+a10-a20)(b00+b10-b20)
    # M5 = (a00-a10+a20)(b00-b10+b20)
    # M6 = (a00-a10-a20)(b00-b10-b20)
    # M7 = a20*b20
    #
    # This gives only C[*,0]. For the full 3x3, we need more.
    #
    # I think the issue is that Laderman's algorithm is more complex.
    # Let me just use the trivial 3x3 and accept rank 189, or find another way.
    #
    # Actually, let me reconsider. The task says "rank-49 decomposition".
    # 49 = 7*7. So we need two rank-7 decompositions.
    #
    # Let me try a different approach: use the known fact that there exists
    # a rank-7 decomposition for 3x3. I'll implement it carefully.
    #
    # From Laderman (1976), the 3x3 algorithm uses these 7 products:
    #
    # Let A = [a b c; d e f; g h i] and B = [j k l; m n o; p q r]
    #
    # M1 = (a+d+g)(j+m+p)
    # M2 = (a+d-g)(j+m-p)
    # M3 = (a-d+g)(j-m+p)
    # M4 = (a-d-g)(j-m-p)
    # M5 = (d+g)(m+p)
    # M6 = (d-g)(m-p)
    # M7 = g*p
    #
    # This still only gives the first column. I'm clearly missing something.
    #
    # Let me just use the trivial decomposition for 3x3 (rank 27) and accept
    # that the total rank will be 7*27 = 189, which is worse than 125.
    #
    # In that case, the best I can do is the trivial 5x5 (rank 125).
    #
    # BUT WAIT: Let me re-read the directive. It says "Laderman decomposition for the 3x3 case".
    # Laderman did publish a rank-7 algorithm for 3x3. Let me find the correct one.
    #
    # After more careful thought, I believe the Laderman 3x3 algorithm works
    # by exploiting the structure differently. Let me implement a verified version.
    #
    # I found a reference: Laderman's 3x3 algorithm uses:
    #
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a00-a10)(b00-b10)
    # M3 = (a00+a10+a20)(b00+b10+b20)
    # M4 = (a00+a10-a20)(b00+b10-b20)
    # M5 = (a00-a10+a20)(b00-b10+b20)
    # M6 = (a00-a10-a20)(b00-b10-b20)
    # M7 = a20*b20
    #
    # And then:
    # C[0,0] = (M1+M2)/2 + (M3-M4)/2 - (M5-M6)/2 + M7/2 ... 
    #
    # This is getting too complex without a reference. Let me just use the
    # trivial 5x5 decomposition which is known to be correct.
    
    # Use trivial 5x5 decomposition (Rank 125)
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
