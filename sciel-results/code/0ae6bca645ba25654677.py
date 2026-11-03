import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    # Rank 16 decomposition for 2x3 by 3x3
    # A is 2x3, B is 3x3, C is 2x3
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    # Block decomposition:
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2] (2x1)
    # B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2] (2x1), B21 = B[2, 0:2] (1x2), B22 = B[2, 2] (1x1)
    
    # C1 = A1 @ B11 + A2 @ B21
    # C2 = A1 @ B12 + A2 @ B22
    
    # C1 is 2x2, C2 is 2x1.
    # C = [C1, C2]
    
    # Products for A1 @ B11 (Strassen, rank 7):
    # P1 = (a11 + a22) * (b11 + b22)
    # P2 = (a21 + a22) * b11
    # P3 = a11 * (b12 - b22)
    # P4 = a22 * (b21 - b11)
    # P5 = (a11 + a12) * b22
    # P6 = (a21 - a11) * (b11 + b12)
    # P7 = (a12 - a22) * (b21 + b22)
    
    # Products for A1 @ B12 (rank 4):
    # Q1 = a11 * b13
    # Q2 = a12 * b23
    # Q3 = a21 * b13
    # Q4 = a22 * b23
    # Result:
    # R13 = Q1 + Q2
    # R23 = Q3 + Q4
    
    # Products for A2 @ B21 (rank 4? No, we found rank 5 for combined A2 part)
    # A2 = [a13, a23]^T
    # B21 = [b31, b32]
    # B22 = b33
    # Terms needed:
    # S11 = a13*b31
    # S12 = a13*b32
    # S21 = a23*b31
    # S22 = a23*b32
    # S13 = a13*b33
    # S23 = a23*b33
    
    # We can use 5 products:
    # T1 = a13 * b31
    # T2 = a23 * b32
    # T3 = (a13 + a23) * (b31 + b32)
    # T4 = (a13 + a23) * b33
    # T5 = a13 * b32 - a23 * b31  (to separate S12 and S21)
    # Then:
    # S11 = T1
    # S22 = T2
    # S12 + S21 = T3 - T1 - T2
    # S12 - S21 = T5
    # So S12 = (T3 - T1 - T2 + T5) / 2
    # S21 = (T3 - T1 - T2 - T5) / 2
    # S13 = T4 - a23*b33? No.
    # T4 = a13*b33 + a23*b33 = S13 + S23.
    # We need S13 and S23 separately.
    # We need one more product?
    # T6 = a13 * b33?
    # Then S13 = T6, S23 = T4 - T6.
    # So 6 products for A2 part.
    # Total 7 + 4 + 6 = 17.
    
    # Can we reduce A2 part to 5?
    # If we use T5 = a13*b33 - a23*b33?
    # Then S13 = (T4 + T5)/2, S23 = (T4 - T5)/2.
    # Yes!
    # So T5 = a13*b32 - a23*b31 is not enough for b33.
    # We need a product involving b33.
    # Let's replace T5 with something that helps both?
    # No, b32 and b33 are different variables.
    # We need to separate S12, S21 (involving b31, b32) and S13, S23 (involving b33).
    # The b31, b32 part needs 3 products (T1, T2, T3) to get S11, S22, S12+S21.
    # We still need S12-S21. That's T5.
    # The b33 part needs 2 products (T4, T6) to get S13, S23.
    # Total 5 + 2 = 7?
    # Wait.
    # S11, S22 are T1, T2.
    # S12+S21 is T3 - T1 - T2.
    # S12-S21 is T5.
    # S13+S23 is T4.
    # S13-S23 is T6.
    # Total 6 products for A2 part.
    # Total 7 + 4 + 6 = 17.
    
    # My previous calculation of 16 was based on A2 part being 5.
    # Is it possible to do A2 part in 5?
    # We have 6 outputs.
    # Can we combine outputs? No.
    # So rank 17 is the safe bet for this block structure.
    
    # Let's check if we can share between A1 and A2 parts.
    # A1 uses a11, a12, a21, a22.
    # A2 uses a13, a23.
    # No overlap in A factors.
    # B factors:
    # B11 uses b11, b12, b21, b22.
    # B12 uses b13, b23.
    # B21 uses b31, b32.
    # B22 uses b33.
    # No overlap in B factors either.
    # So the ranks add up.
    # Rank 7 + 4 + 6 = 17.
    
    # Can we improve A1 @ B12 (rank 4)?
    # Maybe combine with A1 @ B11?
    # No, B factors are disjoint.
    
    # So rank 17 is the result of this decomposition.
    # Schoolbook is 18.
    # 17 is an improvement.
    
    # Let's implement rank 17.
    
    rank = 17
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(6)]
    
    # Indices:
    # A: 0:a11, 1:a12, 2:a13, 3:a21, 4:a22, 5:a23
    # B: 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23, 6:b31, 7:b32, 8:b33
    # C: 0:c11, 1:c12, 2:c13, 3:c21, 4:c22, 5:c23
    
    # Strassen for A1 B11 (cols 0,1 of A, rows 0,1 of B)
    # P1 = (a11 + a22)(b11 + b22)
    # P2 = (a21 + a22)b11
    # P3 = a11(b12 - b22)
    # P4 = a22(b21 - b11)
    # P5 = (a11 + a12)b22
    # P6 = (a21 - a11)(b11 + b12)
    # P7 = (a12 - a22)(b21 + b22)
    
    # U indices for A1: 0, 1, 3, 4
    # V indices for B11: 0, 1, 3, 4
    
    # P1
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[4][0] += 1
    # C11 = P1 - P2 - P4
    # C12 = P3 + P6? No.
    # Standard Strassen:
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # Wait, let's use the standard formulas.
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # Check C12: a11 b12 + a12 b22.
    # P3 + P5 = a11 b12 - a11 b22 + a11 b22 + a12 b22 = a11 b12 + a12 b22. Correct.
    # Check C21: a21 b11 + a22 b21.
    # P2 + P4 = a21 b11 + a22 b11 + a22 b21 - a22 b11 = a21 b11 + a22 b21. Correct.
    # Check C11: a11 b11 + a12 b21.
    # P1 - P2 - P4 = a11 b11 + a11 b22 + a22 b11 + a22 b22 - (a21 b11 + a22 b11) - (a22 b21 - a22 b11)
    # = a11 b11 + a11 b22 + a22 b22 - a21 b11 - a22 b21 + a22 b11
    # This is not C11.
    
    # Let's use the correct Strassen formulas.
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # Wait, P6 and P7 are not used in this standard set?
    # There are different Strassen variants.
    # The one above uses 5 products for 2x2? No, 7.
    # The formulas for C11, C12, C21, C22 using P1..P5 are not complete.
    # P6 and P7 are used in some variants.
    # Let's use a verified rank 7 decomposition.
    
    # Verified Strassen:
    # P1 = (a11 + a22) * (b11 + b22)
    # P2 = (a21 + a22) * b11
    # P3 = a11 * (b12 - b22)
    # P4 = a22 * (b21 - b11)
    # P5 = (a11 + a12) * b22
    # P6 = (a21 - a11) * (b11 + b12)
    # P7 = (a12 - a22) * (b21 + b22)
    
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # This only uses 5 products? No, P6 and P7 are defined but not used?
    # Ah, the standard Strassen uses 7 products to compute 4 entries.
    # The formulas above are for a specific variant.
    # Let's check C11 again.
    # If I use only P1-P5, is it correct?
    # P1 - P2 - P4 = a11 b11 + a11 b22 + a22 b11 + a22 b22 - a21 b11 - a22 b11 - a22 b21 + a22 b11
    # = a11 b11 + a11 b22 + a22 b22 - a21 b11 - a22 b21
    # This is not a11 b11 + a12 b21.
    
    # Correct Strassen formulas:
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # Wait, I am missing terms involving a12 and a21 in C11 and C22?
    # C11 = a11 b11 + a12 b21.
    # The formula gives terms with a21, a22, b22.
    # So this variant is wrong or I am misremembering.
    
    # Let's use the Wikipedia Strassen algorithm.
    # P1 = (a11 + a22) * (b11 + b22)
    # P2 = (a21 + a22) * b11
    # P3 = a11 * (b12 - b22)
    # P4 = a22 * (b21 - b11)
    # P5 = (a11 + a12) * b22
    # P6 = (a21 - a11) * (b11 + b12)
    # P7 = (a12 - a22) * (b21 + b22)
    
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # This is the same. Why is it not working?
    # Ah, P6 and P7 are not used in the final sum?
    # No, that's impossible.
    # Actually, the Wikipedia formulas are:
    # C11 = P1 - P2 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 - P5
    
    # Let's re-evaluate P1 - P2 - P4.
    # P1 = a11 b11 + a11 b22 + a22 b11 + a22 b22
    # P2 = a21 b11 + a22 b11
    # P4 = a22 b21 - a22 b11
    # P1 - P2 - P4 = a11 b11 + a11 b22 + a22 b22 - a21 b11 - a22 b21
    # This is definitely not C11.
    
    # I will use the schoolbook for the 2x2 part to be safe, which is rank 4.
    # Total rank 4 + 4 + 6 = 14?
    # No, schoolbook for 2x2 is 4.
    # So 4 + 4 + 6 = 14.
    # Schoolbook for whole thing is 18.
    # 14 is better.
    
    # Let's implement rank 14.
    # A1 B11: 4 products (a11 b11, a12 b21, a21 b11, a22 b22)
    # A1 B12: 4 products (a11 b13, a12 b23, a21 b13, a22 b23)
    # A2 B2*: 6 products
    
    # Total 14.
    
    rank = 14
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(6)]
    
    # A1 B11
    # C11 += a11 b11
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # C12 += a12 b21
    U[1][1] = 1; V[3][1] = 1; W[1][1] = 1
    # C21 += a21 b11
    U[3][2] = 1; V[0][2] = 1; W[3][2] = 1
    # C22 += a22 b22
    U[4][3] = 1; V[4][3] = 1; W[4][3] = 1
    
    # A1 B12
    # C13 += a11 b13
    U[0][4] = 1; V[2][4] = 1; W[2][4] = 1
    # C13 += a12 b23
    U[1][5] = 1; V[5][5] = 1; W[2][5] = 1
    # C23 += a21 b13
    U[3][6] = 1; V[2][6] = 1; W[5][6] = 1
    # C23 += a22 b23
    U[4][7] = 1; V[5][7] = 1; W[5][7] = 1
    
    # A2 B2*
    # S11 = a13 b31
    U[2][8] = 1; V[6][8] = 1; W[0][8] = 1
    # S22 = a23 b32
    U[5][9] = 1; V[7][9] = 1; W[4][9] = 1
    # S12+S21 = (a13+a23)(b31+b32) - S11 - S22
    U[2][10] = 1; U[5][10] = 1
    V[6][10] = 1; V[7][10] = 1
    # W for C12 (idx 1) and C21 (idx 3)
    # We need to subtract S11 and S22 from the sum.
    # C12 += S12, C21 += S21.
    # S12+S21 = P_sum.
    # We need S12 and S21 separately.
    # Use T5 = a13 b32 - a23 b31.
    U[2][11] = 1; U[5][11] = -1
    V[7][11] = 1; V[6][11] = 1
    # S12 = (P_sum - S11 - S22 + T5) / 2
    # S21 = (P_sum - S11 - S22 - T5) / 2
    
    # C12 (idx 1)
    W[1][10] = 0.5
    W[1][8] = -0.5
    W[1][9] = -0.5
    W[1][11] = 0.5
    
    # C21 (idx 3)
    W[3][10] = 0.5
    W[3][8] = -0.5
    W[3][9] = -0.5
    W[3][11] = -0.5
    
    # S13+S23 = (a13+a23)b33
    U[2][12] = 1; U[5][12] = 1
    V[8][12] = 1
    # S13-S23 = a13 b33 - a23 b33
    U[2][13] = 1; U[5][13] = -1
    V[8][13] = 1
    # S13 = (S13+S23 + S13-S23) / 2
    # S23 = (S13+S23 - S13-S23) / 2
    
    # C13 (idx 2)
    W[2][12] = 0.5
    W[2][13] = 0.5
    
    # C23 (idx 5)
    W[5][12] = 0.5
    W[5][13] = -0.5
    
    return {"U": U, "V": V, "W": W}
