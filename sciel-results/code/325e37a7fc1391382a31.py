"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank-7 decomposition for 3x3x3 matrix multiplication
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1 for all i,j,k
    
    # Known rank-7 decomposition for 3x3x3 MM tensor
    # Using the standard Strassen-based decomposition
    strassen_3x3x3 = [
        # c=0: diagonal term for index 0
        {'U': [(0,0,1),(1,1,1)], 'V': [(0,0,1),(1,1,1)], 'W': [(0,0,1),(1,1,1)]},
        # c=1: diagonal term for index 1  
        {'U': [(1,1,1),(2,2,1)], 'V': [(1,1,1),(2,2,1)], 'W': [(1,1,1),(2,2,1)]},
        # c=2: diagonal term for index 2
        {'U': [(2,2,1)], 'V': [(2,2,1)], 'W': [(2,2,1)]},
        # c=3: cross term (0,1)
        {'U': [(0,1,1),(1,0,1)], 'V': [(0,1,1),(1,0,1)], 'W': [(0,1,1),(1,0,1)]},
        # c=4: cross term (0,2)
        {'U': [(0,2,1),(2,0,1)], 'V': [(0,2,1),(2,0,1)], 'W': [(0,2,1),(2,0,1)]},
        # c=5: cross term (1,2)
        {'U': [(1,2,1),(2,1,1)], 'V': [(1,2,1),(2,1,1)], 'W': [(1,2,1),(2,1,1)]},
        # c=6: remaining terms
        {'U': [(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1),(2,0,1),(2,1,1),(2,2,1)], 
         'V': [(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1),(2,0,1),(2,1,1),(2,2,1)], 
         'W': [(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1),(2,0,1),(2,1,1),(2,2,1)]},
    ]
    
    # This is not correct. Let me use a verified approach.
    # I'll construct the decomposition numerically and verify.
    
    # For now, use a simpler but valid approach:
    # Use Strassen 2x2x2 (rank 7) for top-left 2x2x2, schoolbook for rest
    # Total: 7 + (80-8) = 79... still worse than 80
    
    # Let me just use schoolbook but see if I can find a better decomposition.
    # Actually, let me implement a proper rank-7 3x3x3 decomposition.
    
    # The correct rank-7 decomposition for 3x3x3 MM:
    # Based on the fact that M_3 = M_2 ⊗ M_1 + correction terms
    # After careful derivation, here is a valid rank-7 decomposition:
    
    # U_c, V_c, W_c as 3x3 matrices for c=0..6
    U3 = np.zeros((7, 3, 3))
    V3 = np.zeros((7, 3, 3))
    W3 = np.zeros((7, 3, 3))
    
    # c=0: covers (0,0,0) and helps with other terms
    U3[0,0,0] = 1; U3[0,1,1] = 1
    V3[0,0,0] = 1; V3[0,1,1] = 1
    W3[0,0,0] = 1; W3[0,1,1] = 1
    
    # c=1: covers (1,1,1)
    U3[1,1,1] = 1
    V3[1,1,1] = 1
    W3[1,1,1] = 1
    
    # c=2: covers (2,2,2)
    U3[2,2,2] = 1
    V3[2,2,2] = 1
    W3[2,2,2] = 1
    
    # c=3: covers off-diagonal (0,1) terms
    U3[3,0,1] = 1; U3[3,1,0] = 1
    V3[3,0,1] = 1; V3[3,1,0] = 1
    W3[3,0,1] = 1; W3[3,1,0] = 1
    
    # c=4: covers off-diagonal (0,2) terms  
    U3[4,0,2] = 1; U3[4,2,0] = 1
    V3[4,0,2] = 1; V3[4,2,0] = 1
    W3[4,0,2] = 1; W3[4,2,0] = 1
    
    # c=5: covers off-diagonal (1,2) terms
    U3[5,1,2] = 1; U3[5,2,1] = 1
    V3[5,1,2] = 1; V3[5,2,1] = 1
    W3[5,1,2] = 1; W3[5,2,1] = 1
    
    # c=6: covers remaining terms
    U3[6,0,0] = 1; U3[6,0,1] = 1; U3[6,0,2] = 1
    U3[6,1,0] = 1; U3[6,1,1] = 1; U3[6,1,2] = 1
    U3[6,2,0] = 1; U3[6,2,1] = 1; U3[6,2,2] = 1
    V3[6,0,0] = 1; V3[6,0,1] = 1; V3[6,0,2] = 1
    V3[6,1,0] = 1; V3[6,1,1] = 1; V3[6,1,2] = 1
    V3[6,2,0] = 1; V3[6,2,1] = 1; V3[6,2,2] = 1
    W3[6,0,0] = 1; W3[6,0,1] = 1; W3[6,0,2] = 1
    W3[6,1,0] = 1; W3[6,1,1] = 1; W3[6,1,2] = 1
    W3[6,2,0] = 1; W3[6,2,1] = 1; W3[6,2,2] = 1
    
    # Verify: sum_c U3[c,i,j] * V3[c,j,k] * W3[c,k,i] should = 1 for all i,j,k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total = 0
                for c in range(7):
                    total += U3[c,i,j] * V3[c,j,k] * W3[c,k,i]
                if abs(total - 1) > 1e-10:
                    # This decomposition is invalid, fall back to schoolbook
                    pass
    
    # Since my hand-crafted decomposition may not be correct, let me use a safer approach:
    # Use schoolbook for the entire 4x4x5 problem but with a slight optimization.
    # Actually, the safest improvement is to just keep schoolbook at rank 80.
    
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
# EVOLVE-BLOCK-END
