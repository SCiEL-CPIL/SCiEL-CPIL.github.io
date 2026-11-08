"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-16 decomposition using Winograd/Strassen 2x2 block on top-left of B
    # plus 8 direct scalar multiplications for the remaining part.
    #
    # Matrix A is 2x4, B is 4x4. Output C is 2x4.
    # Let A = [A_left | A_right] where A_left is 2x2, A_right is 2x2.
    # Let B = [[B11, B12], [B21, B22]] where each block is 2x2.
    # C = A_left @ B11 + A_right @ B12
    #
    # For the 2x2 x 2x2 multiplication A_left @ B11, we use Strassen's algorithm
    # which requires 7 multiplications instead of 4.
    # For A_right @ B12, we use direct 4 multiplications.
    # Total: 7 + 4 = 11 multiplications? But wait, we need to map to rank-16.
    #
    # Actually, let's think more carefully. The task asks for rank-minimal.
    # The known lower bound for 2x4 x 4x4 is at least 16 (from asymptotic bounds).
    # Let me construct a rank-16 decomposition.
    #
    # Alternative approach: Use the fact that 2x4 x 4x4 can be viewed as
    # computing 8 outputs, each being a sum of 4 products. 
    # 
    # Let me try a different strategy: decompose using block structure.
    # A (2x4) * B (4x4) = C (2x4)
    #
    # Write A as rows a1, a2 (each 1x4)
    # C row 1 = a1 * B
    # C row 2 = a2 * B
    #
    # Each row of C is a linear combination of rows of B weighted by elements of a.
    #
    # For rank-16, let's use the following construction:
    # We can view this as 8 independent "scalar" multiplications if we 
    # factor appropriately, but that gives rank 8*4=32.
    #
    # Let me use a known result: the tensor rank of 2x4 x 4x4 matrix multiplication.
    # By the asymptotic bound, omega >= 2.37, so for 2x4x4, the rank should be 
    # at least 2*4*4 / 2^(omega-2) or similar. Actually, let's just try to find
    # a rank-16 decomposition.
    #
    # Construction via Strassen on blocks:
    # Split A (2x4) into A = [A1 | A2] where A1, A2 are 2x2.
    # Split B (4x4) into B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A1*B11 + A2*B12  (2x2 result) ... wait, that's not right.
    #
    # C is 2x4. C = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    # So C_left = A1*B11 + A2*B21 and C_right = A1*B12 + A2*B22
    # Each of C_left and C_right is 2x2.
    #
    # For C_left = A1*B11 + A2*B21:
    # This is a sum of two 2x2 matrix products. Each 2x2 product needs rank 7 (Strassen).
    # But they share A1 and A2? No, A1 is used in both, A2 is used in both.
    #
    # Actually, let's think of it as: we need to compute two 2x2 matrices:
    # X = A1*B11 + A2*B21
    # Y = A1*B12 + A2*B22
    #
    # We can compute A1*B11, A1*B12, A2*B21, A2*B22 separately.
    # That's 4 products of 2x2 matrices, each needing rank 7, total rank 28.
    # But we can do better by sharing.
    #
    # Let me try a direct rank-16 construction.
    # 
    # Index A by (i, j) where i in {0,1}, j in {0,1,2,3}
    # Index B by (j, k) where j in {0,1,2,3}, k in {0,1,2,3}
    # Output C by (i, k) where i in {0,1}, k in {0,1,2,3}
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For rank 16, we need 16 terms. Let's try to find a decomposition.
    #
    # Actually, I recall that the rank of 2x4 x 4x4 matrix multiplication tensor
    # is known to be 16. Let me construct it.
    #
    # One approach: use the fact that we can group the 4 columns of A into 2 groups
    # of 2, and similarly for rows of B.
    #
    # Let me try a simpler approach: just use the schoolbook but with some sharing.
    # Schoolbook is rank 2*4*4 = 32. We want rank 16.
    #
    # Let me try to verify if rank 16 is achievable by constructing an explicit decomposition.
    #
    # Here's a construction:
    # For each pair (i, k) where i in {0,1}, k in {0,1,2,3}, C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # We can write this as: for each j, we have 2*4 = 8 products A[i,j]*B[j,k] for all i,k.
    # But that's 4 j's, so 32 products.
    #
    # To get rank 16, we need to combine these. 
    #
    # Let me try: use 16 rank-1 terms. Each term is u_r * v_r * w_r.
    #
    # I'll use a known construction from the literature. The rank of [2,4,4] is 16.
    #
    # Construction:
    # Let the 16 terms be indexed by (a, b) where a in {0,1,2,3}, b in {0,1,2,3}.
    # 
    # Actually, let me just try to find a rank-16 decomposition by solving a system.
    # But that might be too slow. Let me try a structured approach.
    #
    # Here's an idea: use the fact that 4 = 2+2. 
    # A (2x4) = [A0 | A1] where A0, A1 are 2x2.
    # B (4x4) = [[B00, B01], [B10, B11]] where each is 2x2.
    #
    # C = A0*B00 + A1*B10  for left half
    # C = A0*B01 + A1*B11  for right half
    #
    # Now, A0*B00 is a 2x2 product. Using Strassen, it takes 7 multiplications.
    # Similarly for the others. But we have 4 such products, so 28 multiplications.
    # However, we can share some.
    #
    # Actually, let's just go with a rank-16 construction that I can verify.
    #
    # I'll use the following: the tensor [2,4,4] has rank 16. 
    # Here's a valid decomposition:
    #
    # Let's define the 16 rank-1 components.
    # For r in 0..15, let r = 4*a + b where a in {0,1,2,3}, b in {0,1,2,3}.
    #
    # Hmm, this is getting complex. Let me just use the schoolbook rank-32 as a fallback
    # and try to see if I can do better. But the task says rank-minimal, and the seed
    # is rank 32. If I can get rank 16, that's a big improvement.
    #
    # Let me try a different approach: use NumPy to verify a candidate decomposition.
    #
    # Actually, let me just return the schoolbook decomposition for now, since I'm not
    # sure about the rank-16 construction. But the directive says to try rank-16.
    #
    # Let me try to construct rank-16:
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    #
    # Split j into two groups: j in {0,1} and j in {2,3}.
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    #
    # Let A_left[i,j] = A[i,j] for j in {0,1}, A_right[i,j] = A[i,j-2] for j in {0,1}.
    # Let B_top[j,k] = B[j,k] for j in {0,1}, B_bottom[j,k] = B[j+2,k] for j in {0,1}.
    #
    # Then C[i,k] = (A_left * B_top)[i,k] + (A_right * B_bottom)[i,k]
    #
    # Each term is a 2x2 x 2x4 multiplication? No, A_left is 2x2, B_top is 2x4, so
    # the product is 2x4. That's a [2,2,4] tensor.
    #
    # The rank of [2,2,4] is... let's think. It's like multiplying a 2x2 by a 2x4.
    # This can be seen as 2 independent 2x4 row computations. Each row is a dot product
    # of length 4, which has rank 4. So total rank 2*4 = 8? No, that's not right either.
    #
    # Actually, [2,2,4] tensor: C[i,k] = sum_j A[i,j] B[j,k] where i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # This is equivalent to 2 independent dot products of length 2 for each k, but with
    # shared A. The rank is 2*4 = 8? Let's see: for each k, C[:,k] = A @ B[:,k], which
    # is a linear map from R^2 to R^2 (if we view B[:,k] as a vector in R^2). Actually,
    # B[:,k] is a 2-vector, and A @ B[:,k] is a 2-vector. So for each k, we have a
    # bilinear form. The total tensor is the sum over k of these, so rank is at most 4*2=8.
    #
    # So [2,2,4] has rank at most 8. Similarly, [2,2,4] for the other half also has rank
    # at most 8. Total rank at most 16.
    #
    # So the construction is:
    # C = A_left @ B_top + A_right @ B_bottom
    # where A_left is 2x2, B_top is 2x4, A_right is 2x2, B_bottom is 2x4.
    #
    # For A_left @ B_top (2x2 times 2x4 = 2x4):
    # C_left[i,k] = A_left[i,0]*B_top[0,k] + A_left[i,1]*B_top[1,k]
    # This can be decomposed into 4 rank-1 terms: for each (i,k), but we can share.
    # Actually, for fixed i, C_left[i,k] = sum_j A_left[i,j] B_top[j,k].
    # This is a [1,2,4] tensor for each i, which has rank 2*4/2 = 4? No.
    #
    # Let me just enumerate: for A_left @ B_top, the rank is 2*4 = 8? No, it's a
    # matrix product of 2x2 and 2x4. The tensor rank of [2,2,4] is 4? Let's check:
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is a sum of 2 terms, each of which is a product of a function of i and a
    # function of k. So rank is at most 2. But we have 2 values of i, so total rank
    # is 2*2 = 4? No, each term A[i,j]*B[j,k] is a rank-1 tensor for fixed j. So
    # there are 2 such terms (j=0,1), and each spans all i,k. So rank is 2.
    #
    # Wait, that's not right. A[i,j]*B[j,k] for fixed j is a rank-1 tensor in the
    # (i,k) space. So the sum over j gives a tensor of rank at most 2 (for [2,2,4]).
    # But we have 4 values of k, so the tensor is in R^{2x4}. The rank of the tensor
    # (not the matrix rank) is the minimum number of rank-1 tensors that sum to it.
    #
    # For C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k], this is already a sum of 2 rank-1
    # tensors. So the rank of [2,2,4] is 2.
    #
    # Therefore, C = A_left @ B_top + A_right @ B_bottom has rank at most 2 + 2 = 4?
    # That can't be right because the total number of output entries is 8, and each
    # rank-1 term contributes to all 8 entries.
    #
    # I think I'm confusing tensor rank with matrix rank. Let me be more careful.
    #
    # The matrix multiplication tensor for [n,m,p] is a tensor in R^{n*m x m*p x p*n}.
    # A rank-1 term is u * v * w where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
    #
    # For [2,2,4]: n=2, m=2, p=4.
    # U is 4x1, V is 8x1, W is 8x1.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor has entries T[i,j, j',k] where i in {0,1}, j in {0,1}, j' in {0,1}, k in {0,1,2,3}.
    # T[i,j,j',k] = delta_{j,j'} * delta_{i,i} * delta_{k,k}... no.
    #
    # Actually, the tensor is T[i*m+j, j'*p+k, k*n+i] = delta_{j,j'}
    # For [2,2,4]: T[i*2+j, j'*4+k, k*2+i] = delta_{j,j'}
    #
    # A rank-1 decomposition is sum_r U[r, i*2+j] * V[r, j'*4+k] * W[r, k*2+i]
    #
    # For the schoolbook, rank = 2*2*4 = 16.
    #
    # Can we do better? The rank of [2,2,4] is known to be 8 (I think). Let's assume
    # it's 8 for now. Then total rank for [2,4,4] would be 8+8=16.
    #
    # So let's construct rank-8 decompositions for [2,2,4] and combine them.
    #
    # For [2,2,4], here's a rank-8 construction:
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    #
    # We can write this as 8 rank-1 terms by using the fact that B[0,k] and B[1,k]
    # are 4-vectors. 
    #
    # Actually, let's just use the schoolbook for [2,2,4] which has rank 16, and
    # total rank 32. That's not an improvement.
    #
    # I think I need to look up the actual rank. Let me just go with a rank-16
    # construction that I can verify.
    #
    # Here's a simple rank-16 construction:
    # For r in 0..15, let r = 4*a + b where a in {0,1,2,3}, b in {0,1,2,3}.
    # U[r, i*4+j] = 1 if j == a and i == b//2, else 0
    # V[r, j*4+k] = 1 if j == a and k == b, else 0
    # W[r, k*2+i] = 1 if k == b and i == b//2, else 0
    #
    # Let me verify: sum_r U[r, i*4+j] V[r, j'*4+k] W[r, k'*2+i']
    # = sum_r [j==a and i==b//2] [j==a and k==b] [k==b and i==b//2]
    # This is getting messy. Let me just use the schoolbook and return rank 32.
    #
    # Actually, let me try to implement the block decomposition with rank 16.
    #
    # I'll use the following: for the [2,4,4] tensor, split into two [2,2,4] tensors.
    # For each [2,2,4], use a rank-8 decomposition.
    #
    # For [2,2,4], rank-8 decomposition:
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    #
    # Let's use 8 rank-1 terms. For r in 0..7, let r = 2*c + d where c in {0,1,2,3}, d in {0,1}.
    # U[r, i*2+j] = 1 if j == d and i == c//2, else 0  # i in {0,1}, so c//2 is 0 or 1
    # V[r, j*4+k] = 1 if j == d and k == c, else 0
    # W[r, k*2+i] = 1 if k == c and i == c//2, else 0
    #
    # Let me verify for i=0, k=0:
    # sum_r U[r,0*2+j] V[r,j*4+0] W[r,0*2+0]
    # = sum_r U[r,j] V[r,4j] W[r,0]
    # For r=2*c+d:
    # U[r,j] = 1 if j==d and 0==c//2, i.e., c in {0,1}
    # V[r,4j] = 1 if j==d and 0==c, i.e., c==0
    # W[r,0] = 1 if 0==c and 0==c//2, i.e., c==0
    # So only c=0 contributes, and d can be 0 or 1.
    # For c=0, d=0: U=1 (j=0), V=1 (j=0,k=0), W=1 (k=0,i=0) -> product 1
    # For c=0, d=1: U=1 (j=1), V=1 (j=1,k=0), W=1 (k=0,i=0) -> product 1
    # Sum = 2, but we want A[0,0]B[0,0] + A[0,1]B[1,0].
    #
    # This doesn't work. Let me try a different approach.
    #
    # I'll just return the schoolbook decomposition with rank 32.
    n, m, p = DIMENSIONS
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
