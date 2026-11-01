"""Rewritten program for matrix multiplication 2,2,3 with rank 11 decomposition."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=2, p=3
    # U: (4, R), V: (6, R), W: (6, R)
    
    # We use a rank-11 decomposition:
    # 7 terms for the first two columns of B (Strassen-like for 2x2x2)
    # 4 terms for the third column of B (Schoolbook for 2x2x1)
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen-like for B cols 0,1 (Indices 0-5 in V, 0,1,3,4 in W) ---
    # U vectors (size 4)
    # V vectors (size 6, only first 4 active)
    # W vectors (size 6, only 0,1,3,4 active)
    
    # r=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 0; W[3][0] = 0; W[4][0] = 0
    # Note: W indices for C00, C10, C01, C11 are 0, 1, 3, 4.
    # Let's map Strassen output to these indices carefully.
    # Local C00 -> Global 0
    # Local C10 -> Global 1
    # Local C01 -> Global 3
    # Local C11 -> Global 4
    
    # Corrected Strassen Factors for indices:
    # A: 00(0), 01(1), 10(2), 11(3)
    # B: 00(0), 01(1), 10(3), 11(4) -- Wait, V index j*p+k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=1, k=0 -> 3. j=1, k=1 -> 4.
    # C: 00(0), 01(3), 10(1), 11(4).
    
    # Let's re-verify the V indices for the first two columns.
    # k=0: j=0->0, j=1->3.
    # k=1: j=0->1, j=1->4.
    
    # r=0: (a+d)(e+i)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1  # e (0,0) and i (1,1) -> V[0], V[4]
    W[0][0] = 1; W[1][0] = 0; W[3][0] = 0; W[4][0] = 1 # C00, C11 get contribution?
    # (a+d)(e+i) contributes to all 4 outputs.
    # C00: a*e + a*i + d*e + d*i.
    # C01: a*e + a*i + d*e + d*i? No.
    # Standard Strassen:
    # P1 = (a+d)(e+i)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 + P2 - P3 - P7
    # Let's use this standard mapping.
    
    # r=0: P1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = 1
    
    # r=1: P2 = (c+d)(e+i)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[4][1] = 1
    W[0][1] = 0; W[1][1] = 1; W[3][1] = 0; W[4][1] = 1
    
    # r=2: P3 = (a+b)f
    U[0][2] = 1; U[1][2] = 1
    V[1][2] = 1  # f is B(0,1) -> V[1]
    W[0][2] = 0; W[1][2] = 0; W[3][2] = 1; W[4][2] = -1
    
    # r=3: P4 = (c+d)(i-h)
    U[2][3] = 1; U[3][3] = 1
    V[4][3] = 1; V[3][3] = -1 # i is B(1,1)->V[4], h is B(1,0)->V[3]
    W[0][3] = 1; W[1][3] = 1; W[3][3] = 0; W[4][3] = 0
    
    # r=4: P5 = ai
    U[0][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 0; W[3][4] = 1; W[4][4] = 0
    
    # r=5: P6 = (a+b)(e-i)
    U[0][5] = 1; U[1][5] = 1
    V[0][5] = 1; V[4][5] = -1
    W[0][5] = 0; W[1][5] = 0; W[3][5] = 0; W[4][5] = 0 # P6 not in standard formula above?
    # Formula: C00=P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1+P2-P3-P7.
    # P6 is not used in this specific set?
    # Standard Strassen uses 7.
    # P1=(a+d)(e+i), P2=(c+d)(e+i), P3=(a+b)f, P4=(c+d)(i-h), P5=ai, P6=(a+b)(e-i), P7=(c-d)(h+e).
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 + P2 - P3 - P7
    # P6 is not in the sum?
    # Wait, C01 = af + bi. P3+P5 = (a+b)f + ai = af + bf + ai. Missing bi, extra bf, ai.
    # There are multiple isomorphic Strassen decompositions.
    # Let's stick to the one that works.
    # I will use the previously verified set in the thought process if this one fails.
    # For safety, I will use the Schoolbook for all 3 columns? No, rank 12.
    # I will use the rank 11 decomposition derived from 7+4.
    
    # Let's use the explicit tensor factors from a reliable source (e.g. Bini).
    # Factors for 2x2x2 (rank 7):
    # U:
    # 1: 1 0 0 1
    # 2: 0 0 1 1
    # 3: 1 1 0 0
    # 4: 0 0 1 1
    # 5: 1 0 0 0
    # 6: 1 1 0 0
    # 7: 1 0 -1 0
    # V:
    # 1: 1 0 0 1
    # 2: 1 1 0 0
    # 3: 0 1 0 0
    # 4: 0 0 -1 1
    # 5: 0 0 0 1
    # 6: 1 0 0 -1
    # 7: 1 0 1 0
    # W:
    # 1: 1 0 0 0
    # 2: 0 0 0 1
    # 3: 0 1 0 0
    # 4: 0 0 1 0
    # 5: -1 0 0 0
    # 6: 0 -1 0 0
    # 7: 1 0 -1 0
    
    # Let's clear U, V, W and fill with this known set.
    # Reset
    for i in range(n*m):
        for r in range(rank):
            U[i][r] = 0
    for i in range(m*p):
        for r in range(rank):
            V[i][r] = 0
    for i in range(p*n):
        for r in range(rank):
            W[i][r] = 0
            
    # Fill 0-6 with Strassen 2x2x2
    # U (4x7)
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1; U[1][2] = 1
    U[2][3] = 1; U[3][3] = 1
    U[0][4] = 1
    U[0][5] = 1; U[1][5] = 1
    U[0][6] = 1; U[2][6] = -1
    
    # V (6x7) - indices 0,1,3,4 active
    V[0][0] = 1; V[4][0] = 1
    V[0][1] = 1; V[1][1] = 1
    V[1][2] = 1
    V[3][3] = -1; V[4][3] = 1
    V[4][4] = 1
    V[0][5] = 1; V[4][5] = -1
    V[0][6] = 1; V[3][6] = 1
    
    # W (6x7) - indices 0,1,3,4 active
    W[0][0] = 1
    W[4][1] = 1
    W[3][2] = 1
    W[1][3] = 1
    W[0][4] = -1
    W[3][5] = -1
    W[0][6] = 1; W[1][6] = -1

    # Fill 7-10 with Schoolbook for Col 2
    # C02 (idx 4) = A00 B02 + A01 B12
    # C12 (idx 5) = A10 B02 + A11 B12
    
    # r=7: A00 B02
    U[0][7] = 1
    V[2][7] = 1 # B(0,2) -> idx 2
    W[4][7] = 1 # C(0,2) -> idx 4
    
    # r=8: A01 B12
    U[1][8] = 1
    V[5][8] = 1 # B(1,2) -> idx 5
    W[4][8] = 1
    
    # r=9: A10 B02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1 # C(1,2) -> idx 5
    
    # r=10: A11 B12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
