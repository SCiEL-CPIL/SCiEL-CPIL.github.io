"""Matrix multiplication 2x3x7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    
    # Strassen 2x2x2 decomposition (rank 7 instead of 8)
    # For 2x2 matrices A, B, C = A*B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # The 2x3x7 problem can be split as:
    # A is 2x3, B is 3x7
    # We can view A as [A1 A2 A3] where each Ai is 2x1
    # And B as [B1 B2 B3] where each Bj is 1x7
    # So C = A*B = sum_{j=1}^{3} Aj * Bj
    # This is already the schoolbook decomposition with rank 2*3*7 = 42
    
    # To use Strassen, we need to group the columns of A and rows of B into 2x2 blocks.
    # Since A is 2x3, we can split it as [A_left A_right] where A_left is 2x2 and A_right is 2x1
    # Similarly B is 3x7, split as [B_top; B_bottom] where B_top is 2x7 and B_bottom is 1x7
    
    # Then A*B = A_left * B_top + A_right * B_bottom
    # The first term A_left*B_top is a 2x2 * 2x7 multiplication
    # We can write B_top = [B1 B2] where B1, B2 are 2x... wait, B_top is 2x7
    # Actually B_top has 2 rows and 7 columns, so it's a 2x7 matrix
    # A_left is 2x2, so A_left * B_top is 2x7
    
    # Let me reconsider. The standard Strassen approach for 2x3x7:
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x7) into B1 (2x7) and B2 (1x7)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x7 -> use Strassen-like decomposition
    # A2*B2: 2x1 * 1x7 -> this is just rank 2*1*7 = 14
    
    # For A1 (2x2) * B1 (2x7):
    # Write B1 = [B1_left B1_right] where B1_left is 2x3 and B1_right is 2x4? No.
    # B1 is 2x7. We can split it into two 2x? blocks.
    # Actually, let's think of it differently.
    
    # A1 is 2x2, B1 is 2x7.
    # A1*B1 = [A1_col1 A1_col2] * B1 = A1_col1 * B1_row... no.
    # A1*B1 where A1 = [[a11,a12],[a21,a22]] and B1 = [[b11,...,b17],[b21,...,b27]]
    # Result C1 = A1*B1 is 2x7.
    # C1[i,k] = a1i*b1k + a2i*b2k (for i=1,2 and k=1..7)
    
    # We can use the Strassen decomposition by treating the 2x2 block multiplication.
    # For each column k of B1, we compute a 2x1 vector: A1 * [b1k, b2k]^T
    # Using Strassen:
    # C1[1,k] = M1[k] + M4[k] - M5[k] + M7[k]
    # C1[2,k] = M1[k] - M2[k] + M3[k] + M6[k]
    # where for each k:
    # M1[k] = (a11+a22)*(b1k+b2k)
    # M2[k] = (a21+a22)*b1k
    # M3[k] = a11*(b2k-b1k)
    # M4[k] = a22*(b1k-b2k)
    # M5[k] = (a11+a12)*b2k
    # M6[k] = (a21-a11)*(b1k+b2k)
    # M7[k] = (a12-a22)*(b1k+b2k)
    
    # So the rank for A1*B1 part is 7 (one per Strassen term, but each term is a scalar
    # multiplied by a 2x7 outer product structure).
    # Actually, each Ms[k] for all k=1..7 forms a rank-1 tensor contribution.
    # The total rank for A1*B1 is 7*7 = 49? No, that's not right.
    
    # Let me think about the tensor decomposition more carefully.
    # The tensor T has entries T[i,j,k] = 1 if (i,j,k) corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    
    # For the full 2x3x7 problem:
    # C[i,k] = sum_{j=1}^{3} A[i,j] * B[j,k] for i=1,2 and k=1..7
    
    # Using the split A = [A1 | A2] (2x2 | 2x1) and B = [B1; B2] (2x7; 1x7):
    # C[i,k] = sum_{j=1}^{2} A1[i,j]*B1[j,k] + A2[i,1]*B2[1,k]
    
    # The second term A2[i,1]*B2[1,k] has rank 2*1*7 = 14 (schoolbook).
    # The first term: for each k, we have a 2x2 * 2x1 multiplication.
    
    # Using Strassen for the 2x2 part:
    # For each k, C1[i,k] = sum_{j=1}^{2} A1[i,j]*B1[j,k]
    # Using the 7 Strassen terms, each term s gives a contribution:
    # U_s[i] * V_s[j] * W_s[k]
    # where U_s is the coefficient vector for A in term s, V_s for B, W_s for C.
    
    # For term s, the contribution to C1[i,k] is U_s[i] * (sum_j V_s[j]*B1[j,k]) * W_s[i,k]?
    # No, it's U_s[i] * (V_s · B1[:,k]) * W_s where W_s is the output coefficient.
    
    # Actually, the rank-1 tensor for term s is:
    # u_s ⊗ v_s ⊗ w_s where:
    # u_s[i] = coefficient of A[i,j] in Ms
    # v_s[j] = coefficient of B[j,k] in Ms  
    # w_s[i',k'] = coefficient in the output C[i',k']
    
    # For Strassen:
    # M1: u=[1,0,1,0] (for A11,A12,A21,A22 -> but A is 2x2 so u=[1,1,1,1]? No)
    
    # Let me be more careful. For A1 (2x2) with entries:
    # A1 = [[a1, a2], [a3, a4]]
    # B1[:,k] = [[b1], [b2]] (2x1 vector)
    # M1 = (a1+a4)(b1+b2)
    # So u for M1 = [1, 0, 0, 1] (coefficients of a1,a2,a3,a4)
    # v for M1 = [1, 1] (coefficients of b1,b2)
    # w for M1: M1 contributes to C1[1,k] with +1 and to C1[2,k] with +1
    # So w = [1, 1] (for output positions (1,k) and (2,k))
    
    # But w should be a vector of length p*n = 7*2 = 14 (for the C1 part)
    # Actually for the full problem, W has shape (p*n, rank) = (14, rank)
    
    # This is getting complex. Let me just implement the schoolbook solution
    # which is guaranteed to work, and verify it.
    
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
