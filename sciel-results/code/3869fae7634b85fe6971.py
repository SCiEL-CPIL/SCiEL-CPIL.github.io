import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Bini et al. 1979 Rank-23 decomposition for 3x3x3
    # U, V, W are 9x23
    U3 = [[0]*23 for _ in range(9)]
    V3 = [[0]*23 for _ in range(9)]
    W3 = [[0]*23 for _ in range(9)]
    
    # 1-3: Diagonals
    U3[0][0] = 1; V3[0][0] = 1; W3[0][0] = 1
    U3[4][1] = 1; V3[4][1] = 1; W3[4][1] = 1
    U3[8][2] = 1; V3[8][2] = 1; W3[8][2] = 1
    
    # 4-6: Off-diagonal symmetric pairs
    # (0,1) and (1,0)
    U3[1][3] = 1; U3[3][3] = 1
    V3[1][3] = 1; V3[3][3] = 1
    W3[1][3] = 1; W3[3][3] = 1
    
    # (1,2) and (2,1)
    U3[5][4] = 1; U3[7][4] = 1
    V3[5][4] = 1; V3[7][4] = 1
    W3[5][4] = 1; W3[7][4] = 1
    
    # (0,2) and (2,0)
    U3[2][5] = 1; U3[6][5] = 1
    V3[2][5] = 1; V3[6][5] = 1
    W3[2][5] = 1; W3[6][5] = 1
    
    # 7-23: The remaining 17 terms
    # This part is critical.
    # I will use a known complete set.
    # Terms 7-12 involve sums of diagonals and off-diagonals.
    
    # Due to the risk of incorrect coefficients for the remaining 17 terms,
    # I will implement a safe rank-46 decomposition using two schoolbook 3x3x3 blocks?
    # No, schoolbook is 9. 2*9=18.
    # Wait, schoolbook for 3x3x3 is rank 9.
    # So 2 blocks of schoolbook 3x3x3 gives rank 18.
    # 18 < 46.
    # 18 < 54.
    # Why did the current program use rank 54?
    # Because it treated 3x3x6 as a single tensor with rank 3*3*6=54.
    # But 3x3x6 is just two 3x3x3 multiplications.
    # The rank of the direct sum of two tensors is the sum of their ranks.
    # Rank(3x3x3) = 9 (schoolbook).
    # So Rank(3x3x6) <= 18 using schoolbook blocks.
    
    # Let's verify this.
    # C = A * B. B = [B1, B2].
    # C1 = A * B1. C2 = A * B2.
    # C1 is 3x3. C2 is 3x3.
    # We can compute C1 using 9 rank-1 tensors.
    # We can compute C2 using 9 rank-1 tensors.
    # Total 18 rank-1 tensors.
    # This is valid and much better than 54.
    
    # Let's implement this.
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1: k=0,1,2
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is 0,1,2
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    # Block 2: k=3,4,5
    # k' = k-3
    for i in range(n):
        for j in range(m):
            for k_prime in range(3):
                k = k_prime + 3
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
