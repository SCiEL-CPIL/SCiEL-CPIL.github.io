import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Use Laderman's algorithm for 2x3 x 3x3 (rank 15) 
    # and schoolbook for 2x3 x 3x4 (rank 24)
    # Total rank: 15 + 24 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 21 rows
    W = [[0] * rank for _ in range(p * n)]  # 14 rows
    
    # Part 1: Laderman's algorithm for 2x3 x 3x3 (columns 0-14)
    # This computes A (2x3) * B1 (3x3) = C1 (2x3)
    # Laderman's 15 multiplications for 2x2x2 extended... 
    # Actually, let me use a known rank-15 decomposition for 2x3 x 3x3
    
    # For 2x3 x 3x3, we can use the following approach:
    # Split into two 2x2 x 2x2 blocks plus cross terms
    # Or use a direct rank-15 construction
    
    # Let me use a simpler approach: schoolbook for both but optimize
    # 2x3 x 3x3 schoolbook = 18, 2x3 x 3x4 schoolbook = 24, total = 42 (same as before)
    
    # Let me try: use rank-15 for 2x3x3, schoolbook 24 for 2x3x4
    # I need to find a valid rank-15 decomposition for 2x3 x 3x3
    
    # Known result: [2,3,3] has rank 15
    # Construction: think of it as [2,2,2] (rank 7) + extra terms
    # Actually, let me just use a verified construction
    
    # Alternative: use the fact that we can do 2x3 x 3x7 in rank 30+ 
    # by using Strassen-like techniques
    
    # For now, let me implement a careful schoolbook-based approach 
    # but try to reduce by sharing computations
    
    # Actually, the simplest improvement: 
    # The 2x3 matrix has 6 entries, 3x7 has 21 entries, output 2x7 has 14 entries
    # Lower bound: max(6, 21, 14) = 21
    # Schoolbook: 42
    # We can likely do better than 42
    
    # Let me use a block decomposition:
    # Write B (3x7) as [B1 | B2] where B1 is 3x3, B2 is 3x4
    # C = A*B = A*B1 + A*B2
    # A*B1: 2x3 x 3x3, use rank-15 (Laderman-style)
    # A*B2: 2x3 x 3x4, use schoolbook rank-24
    # Total: 39
    
    # Rank-15 decomposition for [2,3,3]:
    # Based on Laderman's algorithm extended
    # Let A = [[a11,a12,a13],[a21,a22,a23]]
    # Let B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    
    # I'll use a known construction. The 15 bilinear forms are:
    # m1 = (a11+a21)*(b11+b31)
    # m2 = (a12+a22)*(b12+b32)  
    # m3 = (a13+a23)*(b13+b33)
    # m4 = (a11+a12+a21+a22)*(b11+b12+b31+b32)
    # m5 = (a11+a13+a21+a23)*(b11+b13+b31+b33)
    # m6 = (a12+a13+a22+a23)*(b12+b13+b32+b33)
    # m7 = a11*b21
    # m8 = a12*b22
    # m9 = a13*b23
    # m10 = a21*b21
    # m11 = a22*b22
    # m12 = a23*b23
    # m13 = (a11-a21)*(b11-b31)
    # m14 = (a12-a22)*(b12-b32)
    # m15 = (a13-a23)*(b13-b33)
    
    # This gives rank 15 for [2,3,3]. Let me verify the output reconstruction.
    # C11 = a11*b11 + a12*b21 + a13*b31
    # From the forms above, I need to check if this works out.
    
    # Actually, let me just use a simpler and verified approach.
    # I'll use schoolbook for the full 2x3 x 3x7 but that's 42.
    # 
    # Better: let me just carefully implement the 39-rank solution.
    
    # For [2,3,3] rank 15, here's a verified construction:
    # The 15 terms and their contributions to C[i][j]:
    
    # Let me define the 15 bilinear forms explicitly.
    # Using a standard [2,3,3] rank-15 decomposition:
    
    laderman_15 = [
        # (U_factor, V_factor, W_contribution)
        # Each entry: (list of (row, coeff) for U, list of (row, coeff) for V, list of (row, coeff) for W)
    ]
    
    # This is getting complex. Let me just use schoolbook 42 for now 
    # and note that 39 is achievable but I need the exact coefficients.
    
    # Actually, let me try a different split: 3x7 = 3x5 + 3x2
    # [2,3,5] and [2,3,2]
    # [2,3,2] schoolbook = 12, [2,3,5] schoolbook = 30, total = 42. No improvement.
    
    # 3x7 = 3x4 + 3x3: [2,3,4]=24, [2,3,3]=18 (schoolbook) or 15 (optimal), total = 39 or 36
    # If I can get [2,3,3] in rank 15, total is 39.
    
    # Let me just implement schoolbook for safety, giving 42.
    # The current program already does this.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
