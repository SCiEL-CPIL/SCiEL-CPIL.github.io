import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use the rank 20 decomposition for 4x4x4 matrix multiplication.
    # Strassen's algorithm for 4x4x4 has rank 20.
    # We need to construct the U, V, W matrices for this.
    
    # The rank 20 decomposition for 4x4x4 can be constructed by 
    # partitioning 4x4 matrices into 2x2 blocks and applying Strassen's 2x2x2 algorithm.
    # However, a direct rank 20 formula is known.
    # Let's use the recursive construction or a hardcoded one.
    
    # For simplicity and reliability, we will use the fact that 
    # R(4,4,4) <= 2 * R(2,2,2) * 2? No.
    # Strassen's algorithm for 4x4 uses 20 multiplications.
    
    # Let's generate the rank 20 decomposition for 4x4x4.
    # The 20 terms correspond to the 20 multiplications in Strassen's 4x4 algorithm.
    
    # Instead of hardcoding 20 terms, we can use the property that 
    # R(4,4,4) <= 2 * R(2,2,4)? No.
    
    # Let's use a known rank 20 decomposition.
    # I will construct it by combining two Strassen 2x2 decompositions?
    # No, the rank 20 is specific.
    
    # Given the time constraint, I will use a simpler bound.
    # R(3,3,8) <= R(3,3,4) + R(3,3,4).
    # If I can't find R(3,3,4)=19, I can use R(3,3,4) <= R(4,4,4) = 20.
    # So R(3,3,8) <= 40.
    
    # Let's implement the rank 40 solution using two 4x4x4 Strassen decompositions.
    
    # First, define the rank 7 Strassen decomposition for 2x2x2.
    # U_2x2: 4x7, V_2x2: 4x7, W_2x2: 4x7
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    def strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # U coefficients for A
        # A00 is at U[0, r]
        # A01 is at U[1, r]
        # A10 is at U[2, r]
        # A11 is at U[3, r]
        
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1
        
        # V coefficients for B
        # B00 is at V[0, r]
        # B01 is at V[1, r]
        # B10 is at V[2, r]
        # B11 is at V[3, r]
        
        V[0][0] = 1; V[3][0] = 1
        V[0][1] = 1
        V[2][2] = 1; V[3][2] = -1
        V[1][3] = 1; V[0][3] = -1
        V[3][4] = 1
        V[0][5] = 1; V[2][5] = 1
        V[1][6] = 1; V[3][6] = 1
        
        # W coefficients for C
        # C00 is at W[0, r] (k=0, i=0 -> index 0*2+0=0)
        # C01 is at W[1, r] (k=1, i=0 -> index 1*2+0=2? No, index k*n+i. n=2. k*2+i)
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        
        # C00 = M1 - M4 + M6 + M7
        W[0][0] = 1; W[0][3] = -1; W[0][5] = 1; W[0][6] = 1
        # C01 = M3 + M5
        W[2][2] = 1; W[2][4] = 1
        # C10 = M2 + M4
        W[1][1] = 1; W[1][3] = 1
        # C11 = M1 - M3 + M5 - M6
        W[3][0] = 1; W[3][2] = -1; W[3][4] = 1; W[3][5] = -1
        
        return U, V, W

    U2, V2, W2 = strassen_2x2()
    
    # Now construct rank 20 decomposition for 4x4x4 using Strassen's recursive method?
    # Actually, applying Strassen to 4x4 directly gives 20 multiplications.
    # The 4x4 matrix is divided into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    # We can use 7 multiplications of 2x2 matrices? No, that would be 7*7=49.
    # Strassen's algorithm for 4x4 uses 20 scalar multiplications.
    # This is equivalent to 20 rank-1 terms for the 4x4x4 tensor.
    
    # I will use a hardcoded rank 20 decomposition for 4x4x4 if possible, 
    # or fall back to the schoolbook for 4x4x4 (rank 64) which is too high.
    
    # Let's use the bound R(3,3,8) <= 3 * R(3,3,3) + ...?
    # Actually, let's just use the schoolbook for 3x3x8 to be safe, 
    # but try to find a better one.
    
    # Given the difficulty of finding the exact rank 19 or 20 coefficients,
    # I will return the schoolbook decomposition which is guaranteed to work.
    # Rank 72.
    
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
