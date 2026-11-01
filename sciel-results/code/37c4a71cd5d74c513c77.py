import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # We want to decompose the 2x4 @ 4x8 matrix multiplication tensor.
    # Tensor T[a, b, c] where:
    #   a indexes A (size n*m = 8), a = i*m + j, i in [0,1], j in [0,3]
    #   b indexes B (size m*p = 32), b = j*p + k, j in [0,3], k in [0,7]
    #   c indexes C (size p*n = 16), c = k*n + i, k in [0,7], i in [0,1]
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i, else 0.
    
    # We use a Strassen-like decomposition.
    # Think of B as two 2x4 blocks: B_top = B[0:2, :] and B_bot = B[2:4, :].
    # Actually, let's think of B as a 4x8 matrix. We can split B into 4x4 blocks:
    # B = [[B11, B12], [B21, B22]] where each is 2x4.
    # A is 2x4. Split A into [A1, A2] where each is 2x2.
    # C = A @ B = [A1, A2] @ [[B11, B12], [B21, B22]]
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each of these is a 2x4 matrix.
    
    # So C_left = A1*B11 + A2*B21 (2x4)
    # C_right = A1*B12 + A2*B22 (2x4)
    
    # For each 2x2 @ 2x4 product, we can use Strassen.
    # Strassen for 2x2 @ 2x2 has rank 7.
    # For 2x2 @ 2x4, we can apply Strassen to each pair of columns, giving 2 * 7 = 14 terms.
    # But we have two such products (left and right), so total would be 28.
    
    # However, the directive says: "treat the 2x4 matrix A as a single entity that interacts
    # with the 7 Strassen intermediates of B, resulting in a total rank of 14 (2 rows of A * 7 Strassen terms)"
    
    # Let me reconsider. The idea is to apply Strassen to the 4x4 blocks of B, but share
    # the Strassen intermediates across the two "row groups" of A.
    
    # Actually, let's think differently. We have A (2x4) and B (4x8).
    # Let's split A into two 2x2 blocks: A = [A1, A2]
    # Let's split B into four 2x4 blocks: B = [[B11, B12], [B21, B22]]
    
    # C = A @ B = [A1, A2] @ [[B11, B12], [B21, B22]]
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    
    # For C_left: A1 is 2x2, B11 is 2x4, A2 is 2x2, B21 is 2x4.
    # We can compute A1*B11 using Strassen (rank 7 for 2x2@2x2, but here it's 2x2@2x4).
    # Actually, 2x2 @ 2x4 can be decomposed into 2 * (2x2 @ 2x2) = 2 * 7 = 14 terms.
    # Similarly for A2*B21: 14 terms.
    # That gives 28 for C_left alone. Too many.
    
    # The key insight from the directive: use 7 Strassen intermediates for B, and each
    # interacts with both rows of A. So total rank = 2 * 7 = 14.
    
    # Let me define the Strassen products for the 4x4 "core" of B.
    # Actually, I think the approach is:
    # View the multiplication as: for each of the 7 Strassen terms applied to B's 4x4 blocks,
    # we get a 2x4 intermediate, which then multiplies with the 2x4 A.
    
    # Let me try a different decomposition. 
    # Consider the tensor T[i*m+j, j*p+k, k*n+i].
    # i in {0,1}, j in {0,1,2,3}, k in {0,...,7}.
    
    # We can group by j. For each j, we have terms involving A[i,j], B[j,k], C[k,i].
    # This is like a sum over j of (A[:,j] outer B[j,:] outer C[:,j]^T) in some sense.
    
    # Let me just try to construct 14 terms manually based on Strassen for the 4x4 case.
    
    # Actually, let me try a simpler approach: use the fact that 2x4 @ 4x8 can be seen as
    # two 2x2 @ 2x4 products summed together, and decompose each 2x2 @ 2x4 using Strassen.
    
    # 2x2 @ 2x4: Let A' be 2x2, B' be 2x4.
    # A' = [[a, b], [c, d]], B' = [[b1, b2, b3, b4], [b5, b6, b7, b8]]
    # C' = A' @ B' = [[a*b1+b*b5, a*b2+b*b6, a*b3+b*b7, a*b4+b*b8],
    #                  [c*b1+d*b5, c*b2+d*b6, c*b3+d*b7, c*b4+d*b8]]
    
    # Using Strassen for 2x2 @ 2x2, applied to each pair of columns:
    # For columns (b1,b5) and (b2,b6):
    # M1 = (a+b)*(b1+b5)
    # M2 = (c+d)*(b1+b5)
    # M3 = (a+b)*b5
    # M4 = (c+d)*b1
    # M5 = (a-c)*(b6-b5)
    # M6 = (c+d)*(b6-b1)
    # M7 = (a-c)*(b2-b1)
    
    # C'[0,0] = M1 - M4 + M3
    # C'[1,0] = M4 + M2 - M3
    # C'[0,1] = M1 - M6 + M7
    # C'[1,1] = M6 + M2 - M7
    
    # This uses 7 rank-1 terms for 2x2 @ 2x2.
    # For 2x2 @ 2x4, we need 2 * 7 = 14 terms.
    
    # But we have TWO such products: A1*B11 + A2*B21 and A1*B12 + A2*B22.
    # If we decompose each separately, we get 4 * 14 = 56 terms.
    
    # The directive suggests sharing: total rank 14.
    
    # Let me try to directly construct 14 terms.
    # I'll use the Strassen decomposition for 2x2 @ 2x2 and extend it.
    
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Let me define the 7 Strassen products for a 2x2 @ 2x2 multiplication.
    # For matrices X = [[x1, x2], [x3, x4]] and Y = [[y1, y2], [y3, y4]]:
    # M1 = (x1+x4)(y1+y4)
    # M2 = (x3+x4)y1
    # M3 = x1(y2-y4)
    # M4 = (x1+x2)(y4)
    # M5 = (x3+x4)(y2)
    # M6 = (x1-x3)(y1+y3)
    # M7 = (x2-x4)(y3+y4)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M3 + M6 - M2
    
    # Now, for our problem, let's define:
    # A = [[A11, A12, A13, A14], [A21, A22, A23, A24]]
    # B = [[B11, B12, B13, B14, B15, B16, B17, B18],
    #      [B21, B22, B23, B24, B25, B26, B27, B28],
    #      [B31, B32, B33, B34, B35, B36, B37, B38],
    #      [B41, B42, B43, B44, B45, B46, B47, B48]]
    
    # C = A @ B, C is 2x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me try to use a block Strassen approach.
    # Split A into [A_left, A_right] where A_left = A[:,0:2], A_right = A[:,2:4]
    # Split B into 4x4 blocks: B = [[B11, B12], [B21, B22]] where each is 2x4.
    # B11 = B[0:2, 0:4], B12 = B[0:2, 4:8]
    # B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    
    # C_left = A_left @ B11 + A_right @ B21
    # C_right = A_left @ B12 + A_right @ B22
    
    # For C_left: A_left is 2x2, B11 is 2x4, A_right is 2x2, B21 is 2x4.
    # We can compute A_left @ B11 using 14 rank-1 terms (2x2 @ 2x4).
    # And A_right @ B21 using another 14 terms.
    # Total 28 for C_left. Similarly 28 for C_right. Total 56.
    
    # This doesn't match the desired rank of 14.
    
    # Let me reconsider the directive: "2 rows of A * 7 Strassen terms"
    # Maybe the idea is to apply Strassen to the 4x4 blocks of B, getting 7 intermediates
    # that are 2x4 matrices, and then each intermediate multiplies with the full 2x4 A.
    
    # Actually, I think I need to be more creative. Let me just try to find 14 terms
    # that work.
    
    # Let me try a different approach: use the Kronecker product structure.
    # The 2x4 @ 4x8 multiplication can be seen as (I_2 ⊗ A) @ (B ⊗ I_2) or similar.
    
    # Actually, let me just implement the schoolbook decomposition for now and verify
    # that the framework works. The rank will be 64, but at least it's correct.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
