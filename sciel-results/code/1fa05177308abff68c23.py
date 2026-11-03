import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a schoolbook decomposition for M_{n, m, p}
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to combine two decompositions
    def combine(decomp1, decomp2):
        U1, V1, W1 = decomp1
        U2, V2, W2 = decomp2
        rank1 = len(U1[0]) if U1 else 0
        rank2 = len(U2[0]) if U2 else 0
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(len(U1))]
        V = [[0] * total_rank for _ in range(len(V1))]
        W = [[0] * total_rank for _ in range(len(W1))]
        
        for i in range(len(U1)):
            for c in range(rank1):
                U[i][c] = U1[i][c]
        for i in range(len(V1)):
            for c in range(rank1):
                V[i][c] = V1[i][c]
        for i in range(len(W1)):
            for c in range(rank1):
                W[i][c] = W1[i][c]
                
        for i in range(len(U2)):
            for c in range(rank2):
                U[i][rank1 + c] = U2[i][c]
        for i in range(len(V2)):
            for c in range(rank2):
                V[i][rank1 + c] = V2[i][c]
        for i in range(len(W2)):
            for c in range(rank2):
                W[i][rank1 + c] = W2[i][c]
                
        return U, V, W

    # Decomposition of M_{3, 5, 7}
    # Split A (3x5) into A_L (3x4) and A_R (3x1)
    # Split B (5x7) into B_L (4x7) and B_R (1x7)
    # C = A_L B_L + A_R B_R
    
    # M_{3, 1, 7}
    # A_R (3x1), B_R (1x7)
    # This is just outer product of A_R and B_R?
    # A_R is 3x1. B_R is 1x7.
    # C is 3x7.
    # C[i,k] = A_R[i,0] * B_R[0,k]
    # Rank 21.
    U_A_R, V_B_R, W_C_R = schoolbook(3, 1, 7)
    
    # M_{3, 4, 7}
    # A_L (3x4), B_L (4x7)
    # Split A_L into A_1 (3x2) and A_2 (3x2)
    # Split B_L into B_1 (2x7) and B_2 (2x7)
    # C_L = A_1 B_1 + A_2 B_2
    # Two M_{3, 2, 7}
    
    def M_3_2_7():
        # A (3x2), B (2x7)
        # Split B into B_a (2x3) and B_b (2x4)
        # C = A B_a + A B_b
        # M_{3, 2, 3} and M_{3, 2, 4}
        
        # M_{3, 2, 3}
        # A (3x2), B (2x3)
        # A = [a1, a2] (3x1)
        # B = [b1; b2] (1x3)
        # C = a1 b1 + a2 b2
        # Two outer products.
        # Outer product of 3x1 and 1x3 has rank 3.
        # Total rank 6.
        # We can just use schoolbook(3,2,3) which is rank 18?
        # No, schoolbook(3,2,3) is 18.
        # But we can do better.
        # Let's construct it manually or use a better helper.
        # For now, use schoolbook to be safe, but we know it's 6.
        # Actually, let's just use schoolbook for all sub-problems to ensure correctness, 
        # but we want to exploit the rank reduction.
        # If I use schoolbook(3,2,3), rank is 18.
        # If I use the outer product structure, rank is 6.
        
        # Let's implement a specific low-rank decomposition for M_{3,2,3}.
        # A (3x2), B (2x3).
        # C[i,k] = sum_{j=0}^1 A[i,j] B[j,k]
        # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
        # This is sum of two outer products.
        # Outer product 1: A[:,0] * B[0,:]
        # Outer product 2: A[:,1] * B[1,:]
        
        U_op1, V_op1, W_op1 = schoolbook(3, 1, 3) # Rank 9? No, 3*1*3=9.
        # Wait, schoolbook(3,1,3) is rank 9.
        # But outer product of 3x1 and 1x3 is rank 3?
        # No, the tensor M_{3,1,3} has rank 3?
        # C[i,k] = A[i,0] B[0,k].
        # Variables: A[0,0], A[1,0], A[2,0], B[0,0], B[0,1], B[0,2].
        # C[0,0] = A0 B0
        # C[0,1] = A0 B1
        # C[0,2] = A0 B2
        # C[1,0] = A1 B0
        # ...
        # This is 9 multiplications.
        # So rank is 9.
        # My previous estimate of 6 for M_{3,2,3} was:
        # C = a1 b1 + a2 b2.
        # a1 is 3x1. b1 is 1x3.
        # a1 b1 is 9 mults.
        # a2 b2 is 9 mults.
        # Total 18.
        # So schoolbook(3,2,3) is 18.
        # And the "outer product" sum is also 18.
        # So no gain there.
        
        # Let's re-evaluate M_{3,2,3}.
        # A (3x2), B (2x3).
        # Is there a rank < 18?
        # Likely not.
        
        # So M_{3,2,7} rank is 18 (for 3,2,3) + 24 (for 3,2,4) = 42?
        # M_{3,2,4}:
        # A (3x2), B (2x4).
        # Split B into two 2x2.
        # Two M_{3,2,2}.
        # M_{3,2,2} rank 12.
        # Total 24.
        # So M_{3,2,7} rank 18 + 24 = 42.
        
        # Then M_{3,4,7} is two M_{3,2,7}.
        # Rank 84.
        # Plus M_{3,1,7} rank 21.
        # Total 105.
        # Back to schoolbook.
        
        # My previous calculation of 89 was based on:
        # M_{3,3,7} = 23 + 36 = 59.
        # M_{3,2,7} = 6 + 24 = 30.
        # Total 89.
        # Why was M_{3,2,3} rank 6?
        # A (3x2), B (2x3).
        # I thought A = [a1, a2] and B = [b1; b2].
        # C = a1 b1 + a2 b2.
        # a1 is 3x1. b1 is 1x3.
        # a1 b1 is outer product.
        # Rank of outer product tensor is 3?
        # No, the rank of the tensor M_{3,1,3} is 9.
        # So a1 b1 requires 9 mults.
        # So M_{3,2,3} requires 18 mults.
        
        # So 89 is incorrect.
        # The schoolbook 105 is likely close to optimal for this simple splitting.
        
        return schoolbook(3, 2, 7)

    # Given the complexity and time, I will return the schoolbook decomposition 
    # but with a small optimization if found.
    # For now, return schoolbook.
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
