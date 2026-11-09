"""Matrix multiplication 4x4x7 via block Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # We view A as 4x4, B as 4x7.
    # Split into 2x2 blocks of size 2x2 (for A) and 2x2 (for B, but B is 4x7, so blocks are 2x7? No, standard block split for 4x4*4x7).
    # Let's split A into 4 blocks of 2x2.
    # Let's split B into 4 blocks of 2x7.
    # Result C is 4x7, split into 4 blocks of 2x7.
    #
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2
    # B = [[B11, B12], [B21, B22]] where Bij are 2x7
    #
    # Standard Strassen for 2x2 blocks (size 2):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a product of a 2x2 matrix and a 2x7 matrix.
    # We can further decompose the 2x2 * 2x7 multiplication.
    # For 2x2 * 2x7, we can use 7 independent 2x2 * 2x1 multiplications? No.
    # A (2x2) * B (2x7) = sum_{k=1}^2 A[:,k] B[k,:].
    # This is a tensor of size 2x2x2x7.
    # Rank of 2x2x2 is 7 (Winograd) or 8 (Strassen).
    # Actually, for 2x2x2, the minimal rank is 7.
    # Let's use the 7-term decomposition for 2x2x2.
    #
    # The 7-term decomposition for 2x2x2 (Winograd scheme W4):
    # Inputs: A (2x2), B (2x2)
    # x1 = A11, x2 = A12, x3 = A21, x4 = A22
    # y1 = B11, y2 = B12, y3 = B21, y4 = B22
    #
    # Actually, the standard 7-term scheme for 2x2x2 is:
    # m1 = (A11 + A22) * (B11 + B22)
    # m2 = (A12 + A22) * B11
    # m3 = A11 * (B12 - B22)
    # m4 = A22 * (B21 - B11)
    # m5 = (A11 + A12) * B22
    # m6 = (A21 - A11) * (B11 + B12)
    # m7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # This is exactly the Strassen algorithm for 2x2 matrices!
    # So each 2x2 * 2x7 multiplication can be done with 7 scalar multiplications per column of B?
    # No, B is 2x7. So we have 7 columns, each 2x1.
    # A (2x2) * b_j (2x1) = c_j (2x1).
    # This is a 2x2x2 tensor (A, b, c).
    # We can apply the 7-term scheme to each column independently.
    # So for each column j of B (which is 2x1), we do 7 multiplications.
    # Total multiplications for one 2x2 * 2x7 block = 7 * 7 = 49.
    #
    # Wait, the Strassen algorithm for 2x2 matrices uses 7 multiplications.
    # Here, the "matrices" are 2x2 (A) and 2x1 (b_j).
    # The product is 2x1.
    # The tensor is 2x2 x 2x1 x 2x1.
    # This is equivalent to 7 independent 2x2x1 multiplications? No.
    # A is 2x2. b is 2x1.
    # c1 = A11*b1 + A12*b2
    # c2 = A21*b1 + A22*b2
    # This is a 2x2x2 tensor (indices: row of A, col of A, row of b).
    # The rank of this tensor is 7.
    # So yes, we can use the 7-term scheme.
    #
    # Let's verify the 7-term scheme for 2x2x2:
    # A = [[a11, a12], [a21, a22]]
    # b = [b1, b2]
    # c = [c1, c2]
    #
    # m1 = (a11 + a22) * (b1 + b2)
    # m2 = (a12 + a22) * b1
    # m3 = a11 * (b2 - b1)  # Note: B12 - B22 becomes b2 - b1 here?
    # Let's map carefully.
    # In the 2x2x2 case, B is 2x1, so B11=b1, B21=b2. B12 and B22 are 0? No, B is just a vector.
    # The standard Strassen is for 2x2 * 2x2.
    # If we treat b as a 2x2 matrix with second column 0, it's not efficient.
    #
    # Actually, the tensor for 2x2 * 2x1 is:
    # T[i, j, k] = 1 if i==k (output row i, input row j, vector index k).
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    # T[0,0,0]=1, T[0,1,1]=1
    # T[1,0,0]=1, T[1,1,1]=1
    # This is the same as the 2x2x2 tensor for matrix multiplication, but with one dimension fixed?
    # No, it's exactly the 2x2x2 tensor.
    # So the rank is 7.
    #
    # Let's derive the 7-term scheme for 2x2 * 2x1.
    # We want c1 = a11*b1 + a12*b2
    #          c2 = a21*b1 + a22*b2
    #
    # Using the same 7 terms as Strassen for 2x2x2:
    # Let B be [[b1, 0], [b2, 0]]? No, that's not right.
    #
    # Actually, we can just use the 7-term scheme directly.
    # The 7-term scheme for 2x2x2 is:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # If we set b12=0, b22=0, b11=b1, b21=b2:
    # m1 = (a11+a22)(b1+0) = (a11+a22)b1
    # m2 = (a21+a22)b1
    # m3 = a11(0-0) = 0
    # m4 = a22(b2-b1)
    # m5 = (a11+a12)0 = 0
    # m6 = (a21-a11)(b1+0) = (a21-a11)b1
    # m7 = (a12-a22)(b2+0) = (a12-a22)b2
    #
    # C11 = m1 + m4 - m5 + m7 = (a11+a22)b1 + a22(b2-b1) - 0 + (a12-a22)b2
    #     = a11*b1 + a22*b1 + a22*b2 - a22*b1 + a12*b2 - a22*b2
    #     = a11*b1 + a12*b2. Correct.
    #
    # C21 = m2 + m4 = (a21+a22)b1 + a22(b2-b1) = a21*b1 + a22*b1 + a22*b2 - a22*b1 = a21*b1 + a22*b2. Correct.
    #
    # So we can use 7 multiplications for each 2x2 * 2x1 product.
    #
    # Total rank:
    # We have 4 blocks of 2x2 * 2x7.
    # Each block has 7 columns, so 7 * 7 = 49 multiplications.
    # Total = 4 * 49 = 196.
    #
    # Wait, the Strassen algorithm for 4x4 * 4x7 uses 7 block multiplications of 2x2 * 2x7.
    # Each 2x2 * 2x7 multiplication uses 7 * 7 = 49 scalar multiplications.
    # So total rank = 7 * 49 = 343.
    #
    # Let's re-evaluate.
    # A is 4x4, B is 4x7.
    # Split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], each 2x2.
    # B = [[B11, B12], [B21, B22]], each 2x7.
    #
    # Strassen:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    #
    # Each Mj is a 2x2 * 2x7 product.
    # Each 2x2 * 2x7 product can be decomposed into 7 * 7 = 49 scalar multiplications.
    # So total rank = 7 * 49 = 343.
    #
    # This is better than 4*4*7=112? No, 343 is worse than 112.
    # Wait, 4*4*7 = 112.
    # So Strassen is worse here?
    #
    # Let's check the rank of 2x2 * 2x7.
    # It's a tensor of size 2x2 x 2x7 x 2x7.
    # No, the output is 2x7.
    # So the tensor is 2x2 (A) x 2x7 (B) x 2x7 (C).
    # The rank is the minimal number of scalar multiplications.
    #
    # Actually, we can decompose 2x2 * 2x7 into 7 independent 2x2 * 2x1 products.
    # Each 2x2 * 2x1 has rank 7.
    # So total rank for 2x2 * 2x7 is 7 * 7 = 49.
    #
    # So total rank for 4x4 * 4x7 using Strassen is 7 * 49 = 343.
    #
    # But the schoolbook rank is 4*4*7 = 112.
    # So Strassen is worse.
    #
    # What if we use a different decomposition?
    # We can decompose 4x4 * 4x7 directly.
    # The tensor is 4x4 x 4x7 x 4x7.
    #
    # Let's try to use the 7-term scheme for 2x2x2 on the 4x4 part.
    # No, that doesn't work directly.
    #
    # What if we use the fact that 4x4 * 4x7 can be seen as 7 independent 4x4 * 4x1 products?
    # Each 4x4 * 4x1 has rank 4*4*1 = 16 (schoolbook).
    # Can we do better?
    # The tensor for 4x4 * 4x1 is 4x4 x 4x1 x 4x1.
    # This is equivalent to 4 independent 4x1 * 4x1? No.
    #
    # Actually, the minimal rank of 4x4 * 4x1 is not known to be less than 16.
    #
    # Let's stick with the schoolbook decomposition for now, which has rank 112.
    #
    # Wait, the task says "Find a rank-minimal exact tensor decomposition".
    # The current program has rank 112.
    #
    # Let's check if we can do better.
    #
    # What if we use the 7-term scheme for 2x2x2 on the 4x4 * 4x4 part, and then handle the 7?
    # No, the 7 is in the inner product.
    #
    # Let's try to decompose 4x4 * 4x7 as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # We can group the sum over j into two parts: j=0,1 and j=2,3.
    # C[i,k] = sum_{j=0}^1 A[i,j] B[j,k] + sum_{j=2}^3 A[i,j] B[j,k]
    #
    # Let A1 = A[:, 0:2], A2 = A[:, 2:4].
    # Let B1 = B[0:2, :], B2 = B[2:4, :].
    #
    # C = A1 * B1 + A2 * B2.
    #
    # This doesn't help directly.
    #
    # Let's try to use the 7-term scheme for 2x2x2 on the 4x4 * 4x4 part, but with B being 4x7.
    #
    # Actually, the rank of 4x4 * 4x7 is at most 4*4*7 = 112.
    #
    # Let's check if there's a known decomposition with rank < 112.
    #
    # For 4x4 * 4x4, the minimal rank is 49 (Strassen).
    # For 4x4 * 4x7, it might be 49 * 7 / 4? No.
    #
    # Let's try to construct a decomposition with rank 49 * 7 = 343? No, that's worse.
    #
    # What if we use the 7-term scheme for each of the 4 rows of A?
    # No.
    #
    # Let's stick with the schoolbook decomposition for now, as it's simple and correct.
    #
    # Wait, the prompt says "Find a rank-minimal exact tensor decomposition".
    # The current program has rank 112.
    #
    # Let's check if we can use a better decomposition.
    #
    # Actually, for 4x4 * 4x7, we can use the following:
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k]
    #
    # We can decompose this into 7 independent 4x4 * 4x1 products.
    # Each 4x4 * 4x1 has rank 16.
    # Total rank = 7 * 16 = 112.
    #
    # So the schoolbook decomposition is optimal in this sense.
    #
    # Let's just return the schoolbook decomposition.
    
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
