"""
Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition.

Strategy:
  The naive (schoolbook) rank is 2*3*9 = 54.
  We use the fact that the 3x9 multiplication can be viewed as a 2x3 multiplication
  over an algebra, but more directly, we exploit the structure to reduce rank.

  Key insight: For multiplying a 2x3 matrix A by a 3x9 matrix B:
    C[i,k] = sum_j A[i,j] * B[j,k]

  The standard decomposition has rank 54. However, we can potentially do better
  by using the fact that for fixed i, the 1x3 row of A times the 3x9 matrix B
  is a linear combination of rows of B.

  Actually, let me think more carefully. The tensor is T[i,j,k] = delta for the
  multiplication: C[i,k] = sum_j A[i,j] B[j,k].

  The rank of the matrix multiplication tensor for (n,m,p) is at most n*m*p.
  For 2x3 by 3x9, that's 54.

  Can we do better? The tensor is T in R^{2 x 3 x 9}. Its rank (tensor rank) 
  is what we want to minimize.

  One approach: use the fact that the 3x9 part can be decomposed. 
  Consider that B is 3x9. We can think of it as 3 rows of length 9.
  
  Actually, a known result: the rank of the matrix multiplication tensor 
  M_{n,m,p} is at most n*m*p, and for many cases the exact rank is unknown.
  
  For 2x3 by 3x9: Let's try to find a decomposition with rank < 54.

  Alternative approach: Use the structure that the second matrix has 9 columns.
  We can group the 9 columns into 3 groups of 3, and use a 3x3 multiplication 
  for the column mixing.

  Actually, let me think about it differently. 
  C[i,k] = sum_j A[i,j] B[j,k]
  
  For each column k of C, C[:,k] = A * B[:,k] where B[:,k] is a 3x1 vector.
  This is a 2x3 matrix times a 3x1 vector, giving a 2x1 vector.
  The rank of the 2x3x1 multiplication tensor is 2*3*1 = 6 (schoolbook).
  But we have 9 such columns, so naive is 54.

  However, we can use linearity: if we can express the 9 column operations 
  using fewer independent tensor factors.

  Let me try a different approach: use the fact that we can combine the 
  multiplication across columns.

  Consider: C[i,k] = sum_j A[i,j] B[j,k]
  
  If we let the 9 columns of B be vectors b_1, ..., b_9 in R^3, then
  C[:,k] = A * b_k.
  
  The map b -> A*b is linear from R^3 to R^2. So C[:,k] = L(b_k) where L is a 
  fixed linear map. But L depends on A which is an input.

  So we need: C[i,k] = sum_j A[i,j] B[j,k]. This is bilinear in A and B.

  The tensor is T[i,j,k] = 1 for all valid indices (it's the "all ones" tensor 
  in the sense that each term A[i,j]*B[j,k] contributes).

  Wait, no. The tensor for matrix multiplication M_{n,m,p} has entries:
  T[i,j,k] = 1 if the index j matches (i.e., the inner dimension), 0 otherwise.
  Actually for M_{n,m,p}: T[i,j,k] = delta_{j,j} which is always 1 for valid 
  indices. The tensor is in R^{n x m x p} with T[i,j,k] = 1 for all i,j,k.

  Hmm, that's not right either. Let me reconsider.

  The matrix multiplication tensor M_{n,m,p} is defined such that:
  (A * B)[i,k] = sum_j sum_l A[i,j] B[j,l] ... no wait.

  For 2x3 times 3x9: C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k].
  The tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,8}.
  So T is the all-ones tensor of shape 2x3x9.

  The rank of the all-ones tensor of shape a x b x c is... actually it's 1!
  Because T[i,j,k] = 1 = 1 * 1 * 1, so it's a rank-1 tensor!

  Wait, that can't be right for matrix multiplication. Let me reconsider.

  Oh I see my confusion. The matrix multiplication tensor is NOT the all-ones tensor.
  The tensor T for matrix multiplication M_{n,m,p} satisfies:
  C[i,k] = sum_{j} A[i,j] B[j,k]
  
  This can be written as: C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
  where T[i,j,k] = 1 for all valid i,j,k.

  But that means T is indeed the all-ones tensor, which has rank 1.
  That would mean rank 1, which is clearly wrong.

  I think I'm confusing myself. Let me look at this more carefully.

  The correct formulation: The bilinear map (A,B) -> A*B where A is n x m and B is m x p.
  We want a decomposition: A*B = sum_{r=1}^{R} U_r(A) * V_r(B)
  where U_r is a linear map from A to some space, V_r from B to some space,
  and the product is a rank-1 outer product or something similar.

  Actually, the standard formulation is:
  C[i,k] = sum_r U[i,r] V[k,r] W[r]
  No, that's not right either.

  Let me use the standard tensor rank formulation for matrix multiplication.
  
  The matrix multiplication tensor M_{n,m,p} is an element of R^{n x m x p} 
  (or more precisely, it's a trilinear form). The rank is the minimum R such that:
  M_{n,m,p} = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
  where u_r in R^n, v_r in R^m, w_r in R^p.

  And the entries of M_{n,m,p} are: M[i,j,k] = 1 if the multiplication 
  A[i,j]*B[j,k] contributes to C[i,k], which is always true for valid indices.
  
  So M[i,j,k] = 1 for all i,j,k. This is the all-ones tensor, rank 1.

  But that contradicts known results. The rank of 2x2x2 matrix multiplication 
  is 7, not 1.

  I must be misunderstanding the tensor definition. Let me reconsider.

  Actually, I think the issue is that the tensor for matrix multiplication is NOT 
  simply the all-ones tensor. The correct tensor is:
  
  For multiplying A (n x m) by B (m x p) to get C (n x p):
  C[i,k] = sum_j A[i,j] B[j,k]
  
  The tensor T in R^{n x m x p} has T[i,j,k] = 1 for all valid i,j,k.
  
  But then C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
  
  This is a trilinear form. The rank of this trilinear form is what we're after.
  
  For n=m=p=2: T[i,j,k] = 1 for all i,j,k in {0,1}. This is the all-ones 
  tensor of shape 2x2x2. The rank of the all-ones tensor of shape 2x2x2 is 
  actually 2 (since it can be written as sum of two rank-1 tensors).
  
  Wait, 1 = (1/2)(1,1,1;1,1,1;1,1,1) + ... no.
  
  The all-ones tensor J of shape 2x2x2: J[i,j,k] = 1.
  Can we write J = sum_r u_r ⊗ v_r ⊗ w_r with few terms?
  J = (1/2) * (1,1) ⊗ (1,1) ⊗ (1,1) + ... 
  Actually (1,1)⊗(1,1)⊗(1,1) gives all entries = 1, so J IS rank 1!
  
  This is very confusing. Let me look up the actual definition.

  After more thought: I believe the issue is that the matrix multiplication 
  tensor is NOT the all-ones tensor in the way I was thinking. 

  The correct tensor for matrix multiplication M_{n,m,p} has entries:
  T[i,j,k] = 1 if j is the "matching" index, but since every j matches 
  (the inner dimension), T[i,j,k] = 1 for all i,j,k.

  I think the confusion is that in the standard literature, the matrix 
  multiplication tensor is indeed the all-ones tensor, and its rank is 
  NOT 1. The rank of the all-ones tensor of shape 2x2x2 is 2, not 1.

  Let me verify: J[i,j,k] = 1 for all i,j,k in {0,1}.
  J = u⊗v⊗w would mean J[i,j,k] = u[i]v[j]w[k]. For this to equal 1 for all 
  i,j,k, we need u[i]v[j]w[k] = 1 for all i,j,k. This requires u[i] = c1, 
  v[j] = c2, w[k] = c3 with c1*c2*c3 = 1. So u = (c1, c1), v = (c2, c2), 
  w = (c3, c3). Then u⊗v⊗w has entries c1*c2*c3 = 1 for all entries. 
  So J IS rank 1!

  But this contradicts the known result that the rank of 2x2x2 matrix 
  multiplication is 7. So I must be fundamentally misunderstanding something.

  Let me reconsider. Perhaps the tensor is not the all-ones tensor. 

  Ah, I think I see. The matrix multiplication is:
  C = A * B where A is n x m and B is m x p.
  C[i,k] = sum_j A[i,j] B[j,k].

  The "tensor" here is a trilinear map T: R^n x R^m x R^p -> R given by
  T(a, b, c) = sum_{i,j,k} a[i] b[j] c[k] M[i,j,k]
  where M[i,j,k] is the matrix multiplication tensor.

  For matrix multiplication, M[i,j,k] = 1 for all valid i,j,k.
  So T(a,b,c) = sum_{i,j,k} a[i] b[j] c[k] = (sum_i a[i])(sum_j b[j])(sum_k c[k]).
  This is indeed rank 1.

  But this can't be right because matrix multiplication is not just the product 
  of sums. The issue is that in matrix multiplication, the indices are linked:
  C[i,k] depends on A[i,j] and B[j,k] for the SAME j.

  So the correct tensor should have M[i,j,k] = 1 only when the indices are 
  "compatible". In matrix multiplication A (n x m) * B (m x p), the index j 
  ranges over the inner dimension, and for each (i,k), we sum over j. The 
  tensor entry M[i,j,k] = 1 for all i,j,k because every combination (i,j,k) 
  contributes to the product.

  I think the resolution is that the tensor rank we're talking about is NOT 
  the rank of the all-ones tensor, but rather the rank of the bilinear map 
  (A,B) -> A*B. This is a different notion.

  The bilinear map (A,B) -> A*B can be decomposed as:
  A*B = sum_{r=1}^R U_r(A) * V_r(B)
  where U_r(A) is an n x p matrix (or a scalar, depending on formulation) 
  and V_r(B) is similarly structured.

  The standard schoolbook decomposition uses R = n*m*p terms:
  A*B = sum_{i,j,k} A[i,j] B[j,k] * E_{i,k}
  where E_{i,k} is the n x p matrix with 1 at (i,k) and 0 elsewhere.
  This gives rank n*m*p.

  The question is whether we can do better. For 2x3 by 3x9, schoolbook is 54.

  Given the time constraints and the fact that finding a rank < 54 decomposition 
  is non-trivial (and may not be possible or may require sophisticated methods),
  I'll implement the schoolbook decomposition as a safe baseline, and add a 
  deadline guard.

  However, let me try one more idea: can we use the fact that the second matrix 
  is 3x9 to reduce the rank? 

  Consider that B is 3x9. We can write B = [B1 | B2 | B3] where each Bi is 3x3.
  Then C = A * B = [A*B1 | A*B2 | A*B3].
  Each A*Bi is a 2x3 matrix multiplication, which has rank at most 2*3*3 = 18 
  (schoolbook). So total rank would be 3*18 = 54, same as before.

  Alternatively, we could use a 3x3 matrix multiplication for the "column mixing" 
  part, but I don't see an immediate improvement.

  Given the complexity, I'll stick with the schoolbook decomposition (rank 54) 
  as it's guaranteed to be correct, and add a deadline guard for safety.
"""

import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p for 2x3 by 3x9 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = n * m * p  # 54

    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 54
    V = [[0] * rank for _ in range(m * p)]  # 27 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Check deadline
    if time.time() > deadline:
        # Should not happen for this simple construction, but guard just in case
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
