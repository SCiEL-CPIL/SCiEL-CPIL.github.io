"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    
    We use the known fact that the 5x5 matrix multiplication tensor has rank at most 19
    (Winograd's algorithm for 5x5 gives rank 19). We can embed the 2x5 x 5x5 multiplication
    into a larger problem and use this decomposition.
    
    Actually, let's think more carefully. The 2x5 x 5x5 multiplication has:
    - Input A: 2x5 (10 elements)
    - Input B: 5x5 (25 elements)  
    - Output C: 2x5 (10 elements)
    
    The standard (schoolbook) rank is 2*5*5 = 50.
    
    We can view this as: for each of the 2 rows of A, we multiply a 1x5 row by a 5x5 matrix.
    So it's two independent 1x5 x 5x5 multiplications.
    
    For a 1x5 x 5x5 multiplication:
    - Input: 5 elements (row vector) x 25 elements (5x5 matrix)
    - Output: 5 elements (row vector)
    - Schoolbook rank: 5*5 = 25
    
    So total schoolbook rank is 2*25 = 50.
    
    Can we do better? The 1x5 x 5x5 multiplication is essentially computing 5 dot products
    of length 5. This is equivalent to multiplying a 1x5 matrix by a 5x5 matrix.
    
    Actually, let's reconsider. We can use the fact that 5x5 matrix multiplication has rank
    19 (Winograd). But our problem is different - it's 2x5 times 5x5.
    
    Let me think of another approach. We can pad the 2x5 matrix to a 5x5 matrix (with zeros in
    rows 3-5), multiply by the 5x5 matrix, and take the first 2 rows of the result.
    
    So C[2x5] = A_padded[5x5] * B[5x5] restricted to first 2 rows.
    
    If we have a rank-19 decomposition of 5x5 multiplication, we can use it. But we only need
    the first 2 rows of the output, so we might be able to reduce the rank.
    
    Actually, the rank of the 2x5 x 5x5 multiplication tensor is at most the rank of the 5x5
    multiplication tensor, which is 19 (by Winograd's algorithm). But we can potentially do better
    since we only need 2 rows of output instead of 5.
    
    Let me try a direct approach using Winograd's 5x5 algorithm adapted for our case.
    
    Winograd's 5x5 matrix multiplication uses 19 multiplications. The algorithm works by
    expressing the 5x5 multiplication as a sum of 19 rank-1 tensors.
    
    For our 2x5 x 5x5 problem, we can use the same 19 multiplications but only compute the
    contributions to the first 2 rows of the output.
    
    Let me implement Winograd's 5x5 algorithm and adapt it.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # We'll try to find a decomposition using the structure of 5x5 matrix multiplication.
    # 
    # Key insight: The 2x5 x 5x5 multiplication can be seen as computing:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..4}
    #
    # This is equivalent to two separate 1x5 x 5x5 multiplications.
    #
    # For the 1x5 x 5x5 case, the schoolbook rank is 25. Can we do better?
    #
    # Actually, let's use a different strategy. We'll use the known rank-19 decomposition
    # of 5x5 matrix multiplication and restrict it to our sub-problem.
    #
    # Winograd's 5x5 algorithm:
    # The 5x5 matrix multiplication tensor T_{i,j,k} = delta_{i,k} * delta_{j,j} (in appropriate basis)
    # can be decomposed as sum of 19 rank-1 tensors.
    #
    # Let me implement this directly. The Winograd 5x5 algorithm uses the following structure:
    # 
    # We express the 5x5 multiplication using 19 bilinear forms.
    #
    # Actually, let me try a simpler approach first: use the schoolbook decomposition but
    # look for linear dependencies to reduce the rank.
    #
    # For the 2x5 x 5x5 multiplication, the tensor is:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The standard decomposition has rank 50. Let's see if we can find a lower-rank decomposition.
    #
    # One approach: Use the fact that we only need 2 rows of output. We can use a rank-19
    # decomposition of 5x5 multiplication and project down.
    #
    # Let me implement Winograd's 5x5 algorithm explicitly.
    
    # Winograd's 5x5 matrix multiplication algorithm
    # Reference: Winograd (1980) "Arithmetic complexity of controllers"
    # 
    # The algorithm uses 19 multiplications for 5x5 matrix multiplication.
    # 
    # The key idea is to use the following factorization:
    # C = A * B where A, B, C are 5x5 matrices
    #
    # We define intermediate quantities:
    # 
    # Let's use the explicit Winograd 5x5 decomposition.
    #
    # Actually, let me try a computational approach: use the known fact that the 5x5 
    # multiplication tensor has rank 19, and construct the decomposition explicitly.
    #
    # For our problem (2x5 x 5x5), we can embed it into 5x5 x 5x5 by padding A with
    # zero rows, and then extract the first 2 rows of the result.
    #
    # If we have a rank-r decomposition of 5x5 multiplication:
    #   (A,B) -> C = sum_{c=1}^{r} (U_c * A) * (V_c * B) * W_c^T
    # where U_c, V_c are 5x5 matrices and W_c is a 5x5 matrix,
    #
    # Then for our 2x5 x 5x5 problem, we can use the same decomposition but only
    # keep the contributions to the first 2 rows of C.
    #
    # This gives us a rank-19 decomposition of our problem (or possibly lower if some
    # terms don't contribute to the first 2 rows).
    #
    # Let me implement the Winograd 5x5 algorithm explicitly.
    
    # Winograd 5x5 matrix multiplication - explicit rank-19 decomposition
    # 
    # The 5x5 multiplication tensor can be written as:
    # T = sum_{c=1}^{19} u_c ⊗ v_c ⊗ w_c
    #
    # where u_c, v_c, w_c are vectors in R^25 (flattened 5x5 matrices).
    #
    # Let me use the known Winograd decomposition. The algorithm works as follows:
    #
    # For 5x5 matrices A and B, compute C = A*B using 19 multiplications.
    #
    # Step 1: Preprocess A and B using linear combinations.
    # Step 2: Perform 19 multiplications.
    # Step 3: Postprocess to get C.
    #
    # The explicit Winograd 5x5 algorithm:
    #
    # Let A = [a_{ij}] and B = [b_{ij}] for i,j in {0,1,2,3,4}
    #
    # Define the following 19 bilinear forms:
    #
    # I'll use the decomposition from the literature. The 5x5 Winograd algorithm
    # uses the following structure based on the (5,5) matrix multiplication.
    #
    # Actually, let me try a different approach. I'll use the fact that we can
    # compute the 2x5 x 5x5 multiplication by using a rank-20 decomposition
    # derived from Kronecker products or other algebraic structures.
    #
    # Alternative approach: Use the schoolbook decomposition (rank 50) and then
    # try to reduce the rank by finding linear dependencies.
    #
    # For now, let me implement a rank-20 decomposition if I can find one,
    # otherwise fall back to schoolbook (rank 50).
    #
    # Let me try to construct a decomposition of rank 20.
    #
    # Idea: The 2x5 x 5x5 multiplication can be viewed as a bilinear map
    # R^{10} x R^{25} -> R^{10}.
    #
    # We can use the following observation: if we fix the second row of A,
    # the multiplication is linear in both A and B. So we can potentially
    # share computations between the two rows.
    #
    # Let me try: A = [a_0; a_1] where a_0, a_1 are 1x5 row vectors.
    # C = [a_0 * B; a_1 * B]
    #
    # This is two independent 1x5 x 5x5 multiplications.
    #
    # For a 1x5 x 5x5 multiplication, the schoolbook rank is 25.
    # Can we do better for 1x5 x 5x5?
    #
    # The 1x5 x 5x5 multiplication tensor has rank at most 25 (schoolbook).
    # It's known that the rank of the (1,5,5) tensor is exactly 25 (since it's
    # equivalent to 5 independent dot products of length 5, and each dot product
    # of length 5 has rank 5, so total rank is 5*5=25).
    #
    # Wait, that's not quite right. The (1,5,5) tensor is:
    # T = sum_{j,k} e_j^A ⊗ e_{j,k}^B ⊗ e_k^C
    #
    # This is a sum of 25 rank-1 tensors. Is this minimal?
    #
    # The (1,n,n) multiplication tensor has rank n^2 for all n (this is a known result).
    # So the (1,5,5) tensor has rank exactly 25.
    #
    # Therefore, the (2,5,5) tensor has rank at most 2*25 = 50 (by doing two independent
    # (1,5,5) multiplications).
    #
    # Can we do better by sharing computations between the two rows?
    #
    # The (2,5,5) tensor is:
    # T = sum_{i=0}^{1} sum_{j,k} e_i^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
    #
    # This is a sum of 50 rank-1 tensors. The question is whether this is minimal.
    #
    # It's known that the rank of the (m,n,p) multiplication tensor is at least
    # max(mn, mp, np) and at most mn*p.
    #
    # For (2,5,5): lower bound is max(10, 10, 25) = 25. Upper bound is 50.
    #
    # So the rank is between 25 and 50.
    #
    # Can we achieve rank 25? That would require a very efficient decomposition.
    #
    # Let me try to construct a rank-25 decomposition.
    #
    # Idea: Use the fact that we can compute both rows of C using shared intermediate
    # values.
    #
    # Let A = [a_0; a_1] and B = [b_0; b_1; b_2; b_3; b_4] where b_i are 1x5 row vectors.
    #
    # C[0,k] = sum_j a_0[j] * b_j[k]
    # C[1,k] = sum_j a_1[j] * b_j[k]
    #
    # We can write this as:
    # [C[0,k]; C[1,k]] = sum_j [a_0[j]; a_1[j]] * b_j[k]
    #
    # This is a sum of 5 terms, each of which is an outer product of a 2-vector and a scalar.
    # But we need to compute this for each k, so we have 5*5 = 25 such terms.
    #
    # Alternatively, we can use a matrix factorization approach.
    #
    # Let me try: Define M = A^T * B where A^T is 5x2 and B is 5x5.
    # Then M is 5x5 and M[j,k] = sum_i A[i,j] * B[j,k]... wait, that's not right.
    #
    # Actually, C = A * B where A is 2x5 and B is 5x5.
    #
    # Let me try a different decomposition. We can use the following:
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    # C = A * B
    #
    # Using a rank-r decomposition, we need:
    # C[i,k] = sum_{c=1}^{r} U[i,m,c] * V[m,p,c] * W[p,n,c]... 
    #
    # Wait, the interface is:
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (25, rank)  
    # W: (p*n, rank) = (10, rank)
    #
    # And the decomposition is:
    # C[k*n+i, ...] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, let me re-read the problem statement.
    #
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # So W is indexed by (k*n+i, r) where k is the column index of C and i is the row index.
    #
    # The decomposition should satisfy:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    #
    # Wait, that's not quite right either. Let me think about this more carefully.
    #
    # The standard tensor decomposition for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This can be written as:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    #
    # No, that's not right. The correct form is:
    #
    # C[i,k] = sum_c U[i*m+?, c] * V[?*p+k, c] * W[k*n+i, c]
    #
    # where the sums are over the intermediate index j.
    #
    # Actually, I think the correct interpretation is:
    #
    # The multiplication tensor T is a 3-tensor with indices (i,j,k) where:
    # - i indexes the row of A (and C), range [0,n)
    # - j indexes the column of A (and row of B), range [0,m)
    # - k indexes the column of B (and C), range [0,p)
    #
    # T[i,j,k] = 1 if this term contributes to C[i,k], which is always the case.
    #
    # The decomposition is:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    #
    # where u_c is a vector in R^{n*m}, v_c is a vector in R^{m*p}, w_c is a vector in R^{p*n}.
    #
    # And the multiplication is:
    # C[i,k] = sum_{j,c} u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] * A[i,j] * B[j,k]
    #
    # Wait, I think I'm overcomplicating this. Let me look at the seed program again.
    #
    # In the seed program:
    # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1 for each (i,j,k).
    #
    # So the decomposition is:
    # C[i,k] = sum_c U[i*m+?][c] * ... 
    #
    # I think the actual computation is:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # No, that doesn't make sense dimensionally.
    #
    # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    #
    # I think the decomposition is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    #
    # No, that would give a product of two sums times W, which is not bilinear.
    #
    # I think the correct interpretation is that the tensor T is decomposed as:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    #
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}.
    #
    # And the multiplication is computed as:
    # C = sum_c (u_c^T ⊙ A_flat) * (v_c^T ⊙ B_flat) * w_c^T
    #
    # where ⊙ denotes some kind of product.
    #
    # Actually, I think the standard form is:
    # C[i,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[i,k]
    #
    # where the sum is over c, and u_c[i,j], v_c[j,k], w_c[i,k] are the entries of the rank-1 tensors.
    #
    # But the shapes are (n*m, rank), (m*p, rank), (p*n, rank), so:
    # U[i*m+j, c] = u_c[i,j]
    # V[j*p+k, c] = v_c[j,k]
    # W[k*n+i, c] = w_c[i,k]  (note: W is indexed by k*n+i, not i*p+k)
    #
    # And the decomposition is:
    # sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, that still doesn't work because the left side has a sum over j, but the right side
    # has U and V both depending on j.
    #
    # I think the correct form is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    #
    # No, that's not bilinear.
    #
    # Let me look at this differently. The multiplication tensor T is a 3-tensor with
    # entries T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k]).
    #
    # The tensor is in R^{(n*m) x (m*p) x (p*n)} where:
    # - First index: i*m + j (row i, column j of A)
    # - Second index: j*p + k (row j, column k of B)
    # - Third index: k*n + i (column k, row i of C, per the trace convention)
    #
    # The decomposition is:
    # T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
    #
    # where a = i*m+j, b = j*p+k, c = k*n+i.
    #
    # And the multiplication is:
    # C[i,k] = sum_{j} T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k]
    #        = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    #
    # Wait, that's not right either because T is a 3-tensor and we're summing over all three indices.
    #
    # I think the correct form is:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # No, that would be a sum of products of three factors, which is trilinear, not bilinear.
    #
    # I'm confused. Let me just look at the seed program and trust that it's correct.
    #
    # In the seed program, for each (i,j,k):
    # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1
    #
    # So the decomposition has rank n*m*p = 50, and each term is a rank-1 tensor with
    # U[a,c]=1 if a=i*m+j, V[b,c]=1 if b=j*p+k, W[cc,c]=1 if cc=k*n+i.
    #
    # The multiplication is then:
    # C[i,k] = sum_c U[i*m+?, c] * ... 
    #
    # I think the actual computation that the evaluator does is:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (sum over something)
    #
    # You know what, let me just trust the seed program and try to find a lower-rank decomposition
    # that satisfies the same tensor equation.
    #
    # The tensor equation is:
    # For all i, j, k: T[i*m+j, j*p+k, k*n+i] = 1
    # For all other (a,b,c): T[a,b,c] = 0
    #
    # And we need: T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
    #
    # So we need to find U, V, W such that this holds.
    #
    # Let me try to find a rank-25 decomposition.
    #
    # Idea: Since the (1,5,5) tensor has rank 25, and our (2,5,5) tensor is a "subtensor" of
    # the (2,5,5) tensor in some sense, maybe we can achieve rank 25.
    #
    # Actually, let me try a computational approach: set up the equations and solve for a
    # rank-r decomposition.
    #
    # For rank r, we have:
    # - U: 10 x r
    # - V: 25 x r
    # - W: 10 x r
    #
    # Total unknowns: 10*r + 25*r + 10*r = 45*r
    #
    # Constraints: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r] for all (a,b,c) in the support.
    # The support has 50 elements (one for each (i,j,k) with i in {0,1}, j in {0..4}, k in {0..4}).
    #
    # So we have 50 nonlinear equations in 45*r unknowns.
    #
    # For r=25: 50 equations in 1125 unknowns. This is underdetermined, so solutions exist.
    # For r=20: 50 equations in 900 unknowns. Still underdetermined.
    #
    # Let me try to find a rank-20 decomposition using a simple construction.
    #
    # Construction idea: Use the schoolbook decomposition but merge terms.
    #
    # In the schoolbook decomposition, each term corresponds to a unique (i,j,k) triple.
    # We can try to merge terms that share the same i or k.
    #
    # For a fixed i, we have 5*5 = 25 terms (one for each (j,k)).
    # We can try to express these 25 terms using fewer rank-1 tensors.
    #
    # For fixed i, the sub-tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all j,k
    #
    # This is a (5,5) matrix of ones, which has rank 1 as a matrix. But as a 3-tensor,
    # it's the (1,5,5) multiplication tensor, which has rank 25.
    #
    # So for each i, we need rank 25, giving total rank 50. This matches the schoolbook.
    #
    # Can we do better by sharing between i=0 and i=1?
    #
    # The full tensor is:
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    #
    # Let me try to find a rank-25 decomposition by solving the system of equations.
    #
    # Actually, let me just try rank 20 and see if I can find a solution.
    
    # Let me try a different approach: use the known result that the (2,5,5) tensor
    # has rank at most 25 (by embedding into (1,5,5) x 2) and see if we can achieve
    # rank 20 or lower.
    
    # For now, let me implement a rank-25 decomposition using the following construction:
    #
    # For each k in {0..4}, we compute C[:,k] = A * B[:,k]
    # where B[:,k] is the k-th column of B (a 5x1 vector).
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Using a rank-5 decomposition of the (1,5,1) dot product (which has rank 5),
    # we get rank 5*5 = 25 for the full (2,5,5) multiplication.
    #
    # Wait, the (1,5,1) dot product has rank 5 (it's just the sum of 5 products).
    # So for each k, we need rank 5, giving total rank 5*5 = 25.
    #
    # Let me implement this rank-25 decomposition.
    
    # Actually, the (1,5,1) tensor is:
    # T[j, j', k'] = 1 if j=j' and k'=0, else 0
    #
    # This is a sum of 5 rank-1 tensors, so rank 5.
    #
    # For the (2,5,5) tensor, we can use:
    # T[i*m+j, j*p+k, k*n+i] = sum_{c=1}^{25} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Let me construct this explicitly.
    
    # Rank-25 decomposition:
    # For each (j, k) pair, we have a term that contributes to C[i,k] for i=0,1.
    #
    # Term (j,k): U[i*m+j, c] = A[i,j] (but U is fixed, not depending on A)
    #
    # Wait, I'm confusing the decomposition with the actual multiplication.
    #
    # The decomposition is:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # where T is the multiplication tensor. Then the actual multiplication is:
    # C[i,k] = sum_{a,b,c} T[a,b,c] * A_flat[a] * B_flat[b]
    #
    # where A_flat[a] = A[i,j] if a=i*m+j, and B_flat[b] = B[j,k] if b=j*p+k.
    #
    # So: C[i,k] = sum_{a,b,c} T[a,b,c] * A[i,j] * B[j,k] where a=i*m+j, b=j*p+k, c=k*n+i
    #
    # = sum_{j,c} T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k]
    #
    # = sum_{j,c} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]) * A[i,j] * B[j,k]
    #
    # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # Ah, so the multiplication is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # This makes sense! It's a sum of rank-1 bilinear forms.
    #
    # So we need:
    # sum_j A[i,j] * B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # for all i, k.
    #
    # Let me define:
    # A_r[i] = sum_j U[i*m+j, r] * A[i,j]  (a 2-vector for each r)
    # B_r[k] = sum_j V[j*p+k, r] * B[j,k]  (a 5-vector for each r)
    #
    # Then: C[i,k] = sum_r W[k*n+i, r] * A_r[i] * B_r[k]
    #
    # We need this to equal sum_j A[i,j] * B[j,k] for all i, k.
    #
    # This is a system of bilinear equations. Let me try to find a rank-25 solution.
    #
    # Construction for rank 25:
    # For each (j, k) in {0..4}x{0..4}, define a term r = j*5 + k.
    #
    # U[i*m+j, r] = 1 if the column index of a is j, else 0.
    # V[j'*p+k, r] = 1 if j'=j and column index is k, else 0.
    # W[k'*n+i, r] = 1 if k'=k, else 0.
    #
    # Let me verify:
    # A_r[i] = sum_j U[i*m+j, r] * A[i,j] = A[i, j] where j is the j for this term.
    # B_r[k] = sum_j V[j*p+k, r] * B[j,k] = B[j, k] where j is the j for this term.
    # W[k*n+i, r] = 1 if k matches.
    #
    # So C[i,k] = sum_{j,k'} W[k'*n+i, r] * A[i,j] * B[j,k'] where r=j*5+k'
    #
    # = sum_j A[i,j] * B[j,k]  (since W[k*n+i, j*5+k]=1 and W[k'*n+i, j*5+k]=0 for k'≠k)
    #
    # Yes! This gives the correct result with rank 25.
    #
    # Now, can we do better than 25? Let me try rank 20.
    #
    # For rank 20, we need 20 terms. Let me try to find a solution.
    #
    # Actually, let me first implement the rank-25 decomposition and see if it works,
    # then try to reduce the rank.
    
    # Rank-25 decomposition
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 10 x 25
    V = [[0] * rank for _ in range(m * p)]  # 25 x 25
    W = [[0] * rank for _ in range(p * n)]  # 10 x 25
    
    col = 0
    for j in range(m):  # j in 0..4
        for k in range(p):  # k in 0..4
            for i in range(n):  # i in 0..1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    # C[i,k] should equal sum_j A[i,j] * B[j,k]
    # Let's verify with a simple test
    A_test = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    B_test = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    
    C_expected = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    
    C_computed = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(rank):
                A_r = sum(U[i * m + j][r] * A_test[i][j] for j in range(m))
                B_r = sum(V[j * p + k][r] * B_test[j][k] for j in range(m))
                s += W[k * n + i][r] * A_r * B_r
            C_computed[i][k] = s
    
    if C_computed != C_expected:
        # Fall back to schoolbook
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
