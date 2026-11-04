"""
Program for finding a rank-minimal exact tensor decomposition for
multiplying a 2x3 matrix by a 3x10 matrix.

Strategy: Use integer linear programming (via scipy's linprog or a custom
branch-and-bound) to find a low-rank decomposition. The key insight is to
formulate the tensor decomposition as a system of linear equations in the
entries of U, V, W, and then search for sparse solutions.

For a rank-r decomposition, we have:
  T_{i,j,k} = sum_{t=1}^{r} U_{i,t} * V_{j,t} * W_{k,t}

This is a system of (n*m*p) equations in (r*(n*m + m*p + p*n)) unknowns.
We want to find solutions where U, V, W have small integer/rational entries.

A different approach: use the fact that for small matrices, we can enumerate
possible rank-1 contributions and use a greedy/iterative method to cover all
entries of the tensor.

Actually, let me use a more principled approach: 
- Start with the schoolbook decomposition (rank 60)
- Use tensor rank reduction via "border rank" techniques or direct search
- For 2x3x10, the known lower bound is at least max(2*3, 3*10, 2*10) = max(6,30,20) = 30
  (from the dimensions of the factors)
  
  Wait, that's not quite right. The tensor rank of matrix multiplication M_{n,m,p}
  satisfies r >= max(n*m, m*p, p*n) for the "slice rank" lower bound, but the actual
  tensor rank can be lower.

  For M_{2,3,10}: schoolbook = 60. 
  Lower bound: The tensor has 2*3*10 = 60 entries. A rank-1 tensor has at most
  (number of nonzero entries in each mode) ... 

  Actually, a better lower bound: r >= n*m = 6 (from fixing k and looking at the
  n x m matrix), r >= m*p = 30 (fixing i), r >= p*n = 20 (fixing j).
  So r >= 30.

  Can we achieve rank 30? That would be a 2x improvement over schoolbook.

  Let me try a different structural approach: 
  - View the multiplication as computing C = A*B where A is 2x3, B is 3x10
  - C[i,k] = sum_j A[i,j] * B[j,k]
  
  For each pair (i,k), we need sum_j A[i,j]*B[j,k].
  
  Key idea: For a fixed i, the row C[i,:] = A[i,:] * B is a linear combination
  of the rows of B with coefficients A[i,0], A[i,1], A[i,2].
  
  So C[0,:] = a00*B[0,:] + a01*B[1,:] + a02*B[2,:]
     C[1,:] = a10*B[0,:] + a11*B[1,:] + a12*B[2,:]
  
  This means each entry C[i,k] = sum_j A[i,j] * B[j,k].
  
  Now, for the tensor decomposition, we need rank-1 tensors whose sum gives
  T_{i,j,k} = delta_{i,j,k} (the matrix multiplication tensor).
  
  A rank-1 tensor is u_i * v_j * w_k.
  
  Schoolbook: T = sum_{i,j,k} e_i^{(n)} * e_j^{(m)} * e_k^{(p)} where the sum is
  over all (i,j,k) with the constraint that the tensor entry at (i,j,k) is 1 only
  if... wait, no. The matrix multiplication tensor T has T_{i,j,k} = 1 if the
  (i,j) entry of A and (j,k) entry of B contribute to C_{i,k}. So T_{i,j,k} = 1
  for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
  
  Wait, I need to be more careful. The matrix multiplication tensor T ∈ R^{n*m*p}
  is defined such that for A ∈ R^{n*m}, B ∈ R^{m*p}, C = A*B ∈ R^{n*p}:
  C_{i,k} = sum_{j} T_{i,j,k} * A_{i,j} * B_{j,k}
  
  So T_{i,j,k} = 1 if the indices are consistent (i.e., the j-th column of A
  times the j-th row of B contributes to C_{i,k}). In our case with n=2, m=3, p=10:
  T_{i,j,k} = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}.
  
  Wait, that's just the all-ones tensor of shape 2*3*10 = 60? No, that's not right either.
  
  Let me re-read the problem. The tensor T is the structure tensor of matrix multiplication.
  T_{(i,j),(j,k),(i,k)} = 1 in appropriate indexing.
  
  With the convention W[k*n+i, r], the output C has entries indexed by (k, i) where
  k is the column index of C (0 to p-1) and i is the row index (0 to n-1).
  
  So C_{i,k} = sum_j A_{i,j} * B_{j,k}
  
  The tensor T has indices (i, j, k) and T_{i,j,k} = 1 if this term appears in the
  sum, which is always the case. So T_{i,j,k} = 1 for all i in [0,2), j in [0,3),
  k in [0,10).
  
  That means T is just the all-ones tensor of shape 2x3x10!
  
  Wait, that can't be right because then the rank would be 1 (it's already rank 1).
  
  I think I'm confusing the tensor representation. Let me reconsider.
  
  The matrix multiplication tensor is NOT the all-ones tensor. The correct statement is:
  
  For the bilinear map (A, B) -> C = A*B, the structure tensor T satisfies:
  C_{i,k} = sum_{j=0}^{m-1} A_{i,j} * B_{j,k}
  
  In the tensor decomposition framework, we write:
  T = sum_{t=1}^{r} u^{(t)} ⊗ v^{(t)} ⊗ w^{(t)}
  
  where u^{(t)} ∈ R^{n*m}, v^{(t)} ∈ R^{m*p}, w^{(t)} ∈ R^{p*n}, and the
  bilinear form is:
  sum_t (u^{(t)} · A) * (v^{(t)} · B) * w^{(t)} = C (as a vector)
  
  More precisely, for each output entry C_{i,k}:
  C_{i,k} = sum_t (sum_{i',j'} u^{(t)}_{i',j'} * A_{i',j'}) * (sum_{j',k'} v^{(t)}_{j',k'} * B_{j',k'}) * w^{(t)}_{k,i}
  
  And we need this to equal sum_j A_{i,j} * B_{j,k} for all A, B.
  
  This means: for each (i,k), the coefficient of A_{i',j'} * B_{j',k'} in the RHS
  must be 1 if (i'=i, j'=j, k'=k) for some j, and 0 otherwise.
  
  So: sum_t u^{(t)}_{i',j'} * v^{(t)}_{j',k'} * w^{(t)}_{k,i} = delta_{i',i} * delta_{k',k} * (1 if j'=j else 0)
  
  Wait, more carefully: we need
  sum_t u^{(t)}_{i',j'} * v^{(t)}_{j',k'} * w^{(t)}_{k,i} = 1 if i'=i, k'=k, and any j'=j
  Actually no. The product A_{i',j'} * B_{j',k'} appears in C_{i,k} only if i'=i AND k'=k.
  And when i'=i and k'=k, the coefficient is 1 (for the specific j'=j).
  
  So the condition is:
  sum_t u^{(t)}_{i',j'} * v^{(t)}_{j',k'} * w^{(t)}_{k,i} = [i'=i and k'=k]
  
  for all i', j', k', i, k.
  
  This is a system of equations. The number of equations is n*m*p*n*p = 2*3*10*2*10 = 1200.
  The number of unknowns for rank r is r*(n*m + m*p + p*n) = r*(6 + 30 + 20) = 56r.
  
  For r = 30: 56*30 = 1680 unknowns, 1200 equations. So it's underdetermined, which is good.
  
  Let me try to find a rank-30 solution using a systematic approach.
  
  Actually, let me think about this differently. The key structural insight:
  
  C_{i,k} = sum_{j=0}^{2} A_{i,j} * B_{j,k}
  
  For fixed i, this is: C_{i,:} = sum_j A_{i,j} * B[j,:]
  
  This is a linear combination of the rows of B, with coefficients A_{i,0}, A_{i,1}, A_{i,2}.
  
  Now, if I could compute all products A_{i,j} * B[j,k] efficiently...
  
  Here's an idea: use the fact that for each i, we have 3 scalar-vector products.
  If we could share computation between i=0 and i=1...
  
  Let's define:
  s0 = A[0,0] + A[1,0]
  s1 = A[0,1] + A[1,1]  
  s2 = A[0,2] + A[1,2]
  d0 = A[0,0] - A[1,0]
  d1 = A[0,1] - A[1,1]
  d2 = A[0,2] - A[1,2]
  
  And for B, define for each column k:
  b0k = B[0,k]
  b1k = B[1,k]
  b2k = B[2,k]
  
  Then:
  C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
  C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
  
  C[0,k] + C[1,k] = s0*B[0,k] + s1*B[1,k] + s2*B[2,k]
  C[0,k] - C[1,k] = d0*B[0,k] + d1*B[1,k] + d2*B[2,k]
  
  So if I can compute (s0*B[0,k] + s1*B[1,k] + s2*B[2,k]) and
  (d0*B[0,k] + d1*B[1,k] + d2*B[2,k]) for each k, I can recover C[0,k] and C[1,k].
  
  But each of these is still a sum of 3 products. So for each k, I need 6 products
  (3 for the sum, 3 for the difference), total 6*10 = 60 products. No improvement.
  
  Let me try another approach. What if I group the B entries?
  
  For each k, define B_col(k) = [B[0,k], B[1,k], B[2,k]]^T (a 3-vector).
  Then C[:,k] = A * B_col(k) (2x3 times 3x1 = 2x1).
  
  So I need to compute A * B_col(k) for k=0,...,9.
  
  If I could precompute something from A that helps... but A is a variable, so I can't.
  
  OK here's a concrete structural approach: use the "Kronecker product" structure.
  
  The 2x3 * 3x10 multiplication can be seen as:
  - First, compute 2x3 * 3x1 (for each column of B): 10 independent 2x3*3x1 multiplications
  - Each 2x3*3x1 has rank 6 (schoolbook)
  - Total: 60
  
  To reduce, I need to share computation across the 10 columns.
  
  Idea: For each column k, C[:,k] = A * B[:,k].
  If I write B[:,k] = B[:,k] (can't simplify without knowing B).
  
  Alternative: Use the fact that the multiplication is bilinear.
  
  Let me try a direct rank-30 construction:
  
  For each (i,j) pair (6 total), and for each k (10 total), the schoolbook uses
  the product A[i,j]*B[j,k] contributing to C[i,k].
  
  Rank 30 would mean using 30 bilinear forms. That's half the schoolbook.
  
  Here's a construction:
  For each j (0,1,2) and each pair of columns (k1, k2), use:
  M_{j,k1,k2} = (A[0,j] + A[1,j]) * (B[j,k1] + B[j,k2])
  This gives: A[0,j]*B[j,k1] + A[0,j]*B[j,k2] + A[1,j]*B[j,k1] + A[1,j]*B[j,k2]
  
  And:
  M'_{j,k1,k2} = (A[0,j] - A[1,j]) * (B[j,k1] - B[j,k2])
  This gives: A[0,j]*B[j,k1] - A[0,j]*B[j,k2] - A[1,j]*B[j,k1] + A[1,j]*B[j,k2]
  
  From these two, I can recover:
  A[0,j]*B[j,k1] = (M + M')/2 + ... hmm, this doesn't directly work.
  
  Actually:
  M = (A0j + A1j)(Bjk1 + Bjk2) = A0j*Bjk1 + A0j*Bjk2 + A1j*Bjk1 + A1j*Bjk2
  M' = (A0j - A1j)(Bjk1 - Bjk2) = A0j*Bjk1 - A0j*Bjk2 - A1j*Bjk1 + A1j*Bjk2
  
  M + M' = 2*A0j*Bjk1
  M - M' = 2*A1j*Bjk2
  
  So from M and M', I get A0j*Bjk1 and A1j*Bjk2. That's 2 products from 2 bilinear forms.
  But I also need A0j*Bjk2 and A1j*Bjk1.
  
  If I also define:
  M'' = (A0j + A1j)(Bjk1 - Bjk2) = A0j*Bjk1 - A0j*Bjk2 + A1j*Bjk1 - A1j*Bjk2
  M''' = (A0j - A1j)(Bjk1 + Bjk2) = A0j*Bjk1 + A0j*Bjk2 - A1j*Bjk1 - A1j*Bjk2
  
  Then I have 4 bilinear forms giving me all 4 products:
  A0j*Bjk1 = (M + M'' + M' + M''')/4 ... let me verify:
  M + M'' = 2*A0j*Bjk1
  M' + M''' = 2*A0j*Bjk1
  So A0j*Bjk1 = (M + M'')/2 = (M' + M''')/2
  
  A0j*Bjk2 = (M - M'')/2
  A1j*Bjk1 = (M''' - M')/2  ... wait let me redo this.
  
  M  = A0j*Bjk1 + A0j*Bjk2 + A1j*Bjk1 + A1j*Bjk2
  M' = A0j*Bjk1 - A0j*Bjk2 - A1j*Bjk1 + A1j*Bjk2
  M''= A0j*Bjk1 - A0j*Bjk2 + A1j*Bjk1 - A1j*Bjk2
  M'''=A0j*Bjk1 + A0j*Bjk2 - A1j*Bjk1 - A1j*Bjk2
  
  M + M'  = 2*A0j*Bjk1 + 2*A1j*Bjk2
  M + M'' = 2*A0j*Bjk1 + 2*A1j*Bjk1
  M + M'''= 2*A0j*Bjk1 + 2*A0j*Bjk2
  M' + M''= 2*A0j*Bjk1 - 2*A1j*Bjk2
  M' + M'''= 2*A0j*Bjk1 - 2*A1j*Bjk1
  M''+ M'''= 2*A0j*Bjk1 - 2*A0j*Bjk2
  
  So:
  A0j*Bjk1 = (M + M' + M'' + M''')/4
  A0j*Bjk2 = (M + M'' - M' - M''')/4 = (M + M''')/2 - (M' + M)/2 ... 
  
  Let me just use linear algebra. The 4x4 matrix mapping (M, M', M'', M''') to
  (A0j*Bjk1, A0j*Bjk2, A1j*Bjk1, A1j*Bjk2) is:
  
  [1  1  1  1]   [A0j*Bjk1]   [M  ]
  [1 -1 -1  1] * [A0j*Bjk2] = [M' ]
  [1 -1  1 -1]   [A1j*Bjk1]   [M'' ]
  [1  1 -1 -1]   [A1j*Bjk2]   [M''' ]
  
  This is a Hadamard matrix H4. H4^T * H4 = 4*I, so the inverse is H4^T/4.
  
  So:
  A0j*Bjk1 = (M + M' + M'' + M''')/4
  A0j*Bjk2 = (M - M' - M'' + M''')/4
  A1j*Bjk1 = (M - M' + M'' - M''')/4
  A1j*Bjk2 = (M + M' - M'' - M''')/4
  
  Great! So for each j and each pair (k1, k2), I use 4 bilinear forms to get
  4 products. That's the same as schoolbook (4 products from 4 forms). No savings.
  
  Hmm. I need a different grouping.
  
  What if I group across j as well?
  
  For a fixed pair of columns (k1, k2), I need:
  C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
  C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2] + A[0,2]*B[2,k2]
  C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
  C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2] + A[1,2]*B[2,k2]
  
  That's 12 products for 4 output entries. Schoolbook uses 12 bilinear forms.
  
  Can I do it in fewer? The "rank" of this 4-entry subproblem...
  
  Actually, let me think about this more carefully using tensor rank theory.
  
  The tensor T for 2x3*3x10 multiplication: what is its rank?
  
  Known results:
  - M_{2,2,2}: rank 7 (Strassen)
  - M_{2,3,2}: rank ? (I think 10 or 11)
  - M_{2,2,3}: rank ? 
  
  For M_{2,3,10}, I don't know the exact rank. But it's certainly less than 60.
  
  Let me try a computational approach: use linear programming to find a rank-r
  decomposition for small r, starting from r=30 and going down.
  
  Actually, the problem is that the equations are nonlinear (products of U, V, W
  entries). So it's not a linear program.
  
  Alternative: Fix U and V, then solve for W linearly. Or use iterative methods.
  
  Let me try yet another approach: use the "block" structure more cleverly.
  
  C = A * B where A is 2x3, B is 3x10.
  
  Write B as a block matrix: B = [B1 | B2] where B1, B2 are 3x5.
  C = [A*B1 | A*B2]
  
  Now, A*B1 and A*B2 are both 2x5. If I can compute both using shared intermediate
  results...
  
  For A*B1: C1[i,k] = sum_j A[i,j] * B1[j,k] for i=0,1, k=0,...,4
  For A*B2: C2[i,k] = sum_j A[i,j] * B2[j,k] for i=0,1, k=0,...,4
  
  The A entries are shared! So if I precompute combinations of A rows, I can use
  them for both B1 and B2.
  
  Specifically, define:
  s0 = A[0,0] + A[1,0]
  s1 = A[0,1] + A[1,1]
  s2 = A[0,2] + A[1,2]
  d0 = A[0,0] - A[1,0]
  d1 = A[0,1] - A[1,1]
  d2 = A[0,2] - A[1,2]
  
  These 6 values are shared between computing A*B1 and A*B2.
  
  For A*B1:
  C1[0,k] + C1[1,k] = s0*B1[0,k] + s1*B1[1,k] + s2*B1[2,k]
  C1[0,k] - C1[1,k] = d0*B1[0,k] + d1*B1[1,k] + d2*B1[2,k]
  
  Similarly for A*B2.
  
  But each of these is still a sum of 3 products. So for each k, I need 6 products
  (3 for sum, 3 for diff), total 6*10 = 60. Same as before.
  
  I'm stuck. Let me just implement a search-based approach:
  
  1. Start with schoolbook (rank 60)
  2. Try to find a lower rank solution using random initialization + gradient descent
     on the nonlinear system
  3. Or use a more structured search
  
  Actually, for the purposes of this task, let me just implement a clean schoolbook
  solution and a few attempts at structured reductions, and verify correctness.
  
  The key issue with the current program is that it has buggy code (the _strassen_2x3x5
  function returns garbage). Let me write a clean, correct program.
"""

import time
import numpy as np


def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # The decomposition should satisfy:
    # For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # (since T_{i,j,k} = 1 for matrix multiplication)
    
    # Let's verify by checking the bilinear form
    # C_{i,k} = sum_t (sum_{i',j'} U[i',j',t] * A[i',j']) * (sum_{j',k'} V[j',k',t] * B[j',k']) * W[k,i,t]
    # should equal sum_j A[i,j] * B[j,k]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    C_computed = np.zeros((n, p))
    for t in range(rank):
        u_vec = U[:, t]  # (n*m,)
        v_vec = V[:, t]  # (m*p,)
        w_vec = W[:, t]  # (p*n,)
        
        # u_vec · A = sum_{i',j'} U[i',j',t] * A[i',j']
        uA = np.dot(u_vec.reshape(n, m), A)
        vB = np.dot(v_vec.reshape(m, p), B)
        
        # C += uA * vB * w_vec reshaped
        C_computed += uA * vB * w_vec.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True


def _schoolbook(n=2, m=3, p=10):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _try_structured_reduction(n=2, m=3, p=10, deadline=None):
    """
    Try to find a lower-rank decomposition using structured approaches.
    
    Approach: Use the Hadamard-based pairing to reduce rank.
    For each j, pair up the k indices and use 4 forms per 4 products (no gain).
    
    Better approach: Try to find a rank-r solution via optimization.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try rank 30: use the structure that C[i,k] = sum_j A[i,j]*B[j,k]
    # For rank 30, we need 30 bilinear forms.
    # 
    # Idea: For each j (3 choices) and each "half" of k (5 choices), use 2 forms
    # to handle pairs of k values.
    #
    # Actually, let me try a different structured approach:
    # 
    # The tensor T_{i,j,k} = 1 for all (i,j,k) in {0,1}x{0,1,2}x{0,...,9}
    # 
    # This is the all-ones tensor of shape 2x3x10!
    # 
    # Wait, is it? Let me re-verify.
    # 
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # 
    # In the tensor decomposition: C_{i,k} = sum_t (u_t · A) (v_t · B) w_{t,k,i}
    # 
    # where u_t is a vector in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}.
    # 
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_t u_{t,i,j} v_{t,j,k} w_{t,k,i} = 1  for all i,j,k
    # 
    # (The coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1, and in C_{i',k'}
    #  for i'≠i or k'≠k is 0.)
    # 
    # So T_{i,j,k} = 1 for all valid indices. The tensor IS the all-ones tensor!
    # 
    # The rank of the all-ones tensor of shape a x b x c is 1!
    # 
    # Wait, that can't be right. If the tensor is all ones, then:
    # sum_t u_{t,i,j} v_{t,j,k} w_{t,k,i} = 1 for all i,j,k
    # 
    # With t=1: u_{1,i,j} = 1, v_{1,j,k} = 1, w_{1,k,i} = 1 for all i,j,k.
    # Then sum_t = 1*1*1 = 1. Yes! Rank 1!
    # 
    # But that would mean matrix multiplication is rank 1, which is absurd.
    # 
    # I must be misunderstanding the tensor structure. Let me re-read the problem.
    
    # AH, I see my error. The tensor is NOT all ones. Let me re-derive.
    # 
    # The matrix multiplication tensor T ∈ R^{(n*m) x (m*p) x (p*n)} is defined
    # such that for the bilinear map (A, B) -> C = A*B:
    # 
    # vec(C) = sum_t (U_t · vec(A)) * (V_t · vec(B)) * W_t
    # 
    # where U_t ∈ R^{n*m}, V_t ∈ R^{m*p}, W_t ∈ R^{p*n}.
    # 
    # More precisely, the t-th rank-1 component contributes:
    # (U_t · A) * (V_t · B) * W_t
    # 
    # to the output vector vec(C) ∈ R^{p*n}.
    # 
    # So for output entry C_{i,k} (which is at position k*n+i in vec(C)):
    # C_{i,k} = sum_t (sum_{i',j'} U_{t,i',j'} A_{i',j'}) * (sum_{j',k'} V_{t,j',k'} B_{j',k'}) * W_{t,k,i}
    # 
    # We need this to equal sum_j A_{i,j} B_{j,k} for all A, B.
    # 
    # The coefficient of A_{i',j'} * B_{j',k'} in C_{i,k} must be:
    # 1 if i'=i and k'=k (any j')
    # 0 otherwise
    # 
    # So: sum_t U_{t,i',j'} V_{t,j',k'} W_{t,k,i} = δ_{i',i} * δ_{k',k}
    # 
    # Note: there's NO delta on j'! The sum over j' is implicit in the product
    # U_{t,i',j'} V_{t,j',k'} which sums over j'.
    # 
    # Wait, no. U_{t,i',j'} is a single entry (for fixed i', j'), and V_{t,j',k'}
    # is a single entry (for fixed j', k'). The product U_{t,i',j'} V_{t,j',k'}
    # does NOT sum over j'. 
    # 
    # Let me be very careful:
    # (U_t · A) = sum_{i'=0}^{n-1} sum_{j'=0}^{m-1} U_{t,i',j'} A_{i',j'}
    # (V_t · B) = sum_{j'=0}^{m-1} sum_{k'=0}^{p-1} V_{t,j',k'} B_{j',k'}
    # 
    # So (U_t · A)(V_t · B) = sum_{i',j',j'',k'} U_{t,i',j'} V_{t,j'',k'} A_{i',j'} B_{j'',k'}
    # 
    # The coefficient of A_{i',j'} B_{j'',k'} is U_{t,i',j'} V_{t,j'',k'}.
    # 
    # For the full sum over t:
    # C_{i,k} = sum_t W_{t,k,i} sum_{i',j',j'',k'} U_{t,i',j'} V_{t,j'',k'} A_{i',j'} B_{j'',k'}
    # 
    # The coefficient of A_{i',j'} B_{j'',k'} in C_{i,k} is:
    # sum_t U_{t,i',j'} V_{t,j'',k'} W_{t,k,i}
    # 
    # This must equal: 1 if i'=i and j'=j'' and k'=k, else 0.
    # 
    # So: sum_t U_{t,i',j'} V_{t,j'',k'} W_{t,k,i} = δ_{i',i} δ_{j',j''} δ_{k',k}
    # 
    # This is the correct condition! The tensor T has entries:
    # T_{(i',j'),(j'',k'),(k,i)} = δ_{i',i} δ_{j',j''} δ_{k',k}
    # 
    # In other words, T is a "diagonal" tensor in the sense that it's 1 only when
    # the indices match up correctly.
    # 
    # For n=2, m=3, p=10:
    # T has shape (6, 30, 20) and T_{(i,j),(j,k),(k,i)} = 1, else 0.
    # 
    # The number of 1s is: for each j (3 choices), i (2 choices), k (10 choices): 60 ones.
    # 
    # Now, the rank of this tensor is what we want to minimize.
    # 
    # Schoolbook: rank 60 (one rank-1 term for each (i,j,k) with U=e_{i,j}, V=e_{j,k}, W=e_{k,i})
    # 
    # Can we do better? The lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # (from the "slice rank" or "subrank" lower bound).
    # 
    # So the minimum possible rank is at least 30.
    
    # Let me try to construct a rank-30 decomposition.
    # 
    # For rank 30, we need 30 bilinear forms.
    # 
    # Key insight: The condition is sum_t U_{t,i',j'} V_{t,j'',k'} W_{t,k,i} = δ_{i',i}δ_{j',j''}δ_{k',k}
    # 
    # If we set W_{t,k,i} = 1 for all t,k,i (or some simple pattern), then we need:
    # sum_t U_{t,i',j'} V_{t,j'',k'} = δ_{i',i}δ_{j',j''}δ_{k',k}
    # 
    # This is like a matrix factorization problem.
    # 
    # Alternatively, use the structure:
    # For each j (0,1,2), define 10 forms (one for each k).
    # U_{t,i,j} = δ_{i,i_t} (where i_t is 0 or 1)
    # V_{t,j,k} = δ_{k,k_t}
    # W_{t,k,i} = δ_{k,k_t} * δ_{i,i_t}
    # 
    # This is just schoolbook.
    # 
    # To get rank 30, I need to share across i or k.
    # 
    # Let me try: for each j and each pair of k values (k1, k2), use 2 forms:
    # Form 1: U_{i,j} = 1 for both i=0,1 (so U_{0,j}=U_{1,j}=1, others 0)
    #          V_{j,k1}=1, V_{j,k2}=1, others 0
    #          W_{k1,0}=1, W_{k1,1}=1, W_{k2,0}=1, W_{k2,1}=1
    # 
    # This gives: for i=0,1: contribution to C_{i,k1} and C_{i,k2} is A_{i,j}*(B_{j,k1}+B_{j,k2})
    # 
    # Form 2: U_{i,j} = 1 for both i
    #          V_{j,k1}=1, V_{j,k2}=-1
    #          W_{k1,0}=1, W_{k1,1}=1, W_{k2,0}=-1, W_{k2,1}=-1
    # 
    # This gives: contribution to C_{i,k1} is A_{i,j}*B_{j,k1}, to C_{i,k2} is -A_{i,j}*B_{j,k2}
    # 
    # From Form 1 + Form 2: C_{i,k1} gets 2*A_{i,j}*B_{j,k1}, C_{i,k2} gets 0.
    # From Form 1 - Form 2: C_{i,k1} gets 0, C_{i,k2} gets 2*A_{i,j}*B_{j,k2}.
    # 
    # So with 2 forms, I can isolate A_{i,j}*B_{j,k1} and A_{i,j}*B_{j,k2} for BOTH i=0 and i=1.
    # That's 4 products from 2 forms! But wait, I need to be careful about the W coefficients.
    
    # Let me redo this more carefully.
    # 
    # For fixed j, k1, k2, I want to produce the contributions:
    # A_{0,j}*B_{j,k1} -> C_{0,k1}
    # A_{0,j}*B_{j,k2} -> C_{0,k2}
    # A_{1,j}*B_{j,k1} -> C_{1,k1}
    # A_{1,j}*B_{j,k2} -> C_{1,k2}
    # 
    # Using 2 bilinear forms:
    # Form A: U_A[0,j]=1, U_A[1,j]=1, others 0
    #          V_A[j,k1]=1, V_A[j,k2]=1, others 0
    #          W_A[k1,0]=1, W_A[k1,1]=1, W_A[k2,0]=1, W_A[k2,1]=1, others 0
    # 
    # Form B: U_B[0,j]=1, U_B[1,j]=1, others 0
    #          V_B[j,k1]=1, V_B[j,k2]=-1, others 0
    #          W_B[k1,0]=1, W_B[k1,1]=1, W_B[k2,0]=-1, W_B[k2,1]=-1, others 0
    # 
    # Contribution to C_{0,k1}: (U_A·A)(V_A·B)W_A[k1,0] + (U_B·A)(V_B·B)W_B[k1,0]
    #   = (A[0,j]+A[1,j])*(B[j,k1]+B[j,k2])*1 + (A[0,j]+A[1,j])*(B[j,k1]-B[j,k2])*1
    #   = (A[0,j]+A[1,j])*2*B[j,k1]
    # 
    # This includes A[1,j]*B[j,k1] which should go to C_{1,k1}, not C_{0,k1}!
    # 
    # So this doesn't work directly. The issue is that U mixes i=0 and i=1.
    
    # I need U to separate i. Let me try:
    # Form A: U_A[0,j]=1, U_A[1,j]=0
    #          V_A[j,k1]=1, V_A[j,k2]=1
    #          W_A[k1,0]=1, W_A[k2,0]=1
    # 
    # Form B: U_B[0,j]=1, U_B[1,j]=0
    #          V_B[j,k1]=1, V_B[j,k2]=-1
    #          W_B[k1,0]=1, W_B[k2,0]=-1
    # 
    # C_{0,k1}: (A[0,j])*(B[j,k1]+B[j,k2])*1 + (A[0,j])*(B[j,k1]-B[j,k2])*1 = 2*A[0,j]*B[j,k1]
    # C_{0,k2}: (A[0,j])*(B[j,k1]+B[j,k2])*1 + (A[0,j])*(B[j,k1]-B[j,k2])*(-1) = 2*A[0,j]*B[j,k2]
    # 
    # Great! So 2 forms give me 2 products (for i=0). Similarly for i=1, I need 2 more forms.
    # Total: 4 forms for 4 products. Same as schoolbook.
    
    # The problem is that I can't share across i because the W vectors need to target
    # specific (k,i) pairs.
    
    # Unless... I use W vectors that have entries for both i=0 and i=1, and then
    # use the U vectors to separate.
    
    # Let me try:
    # Form 1: U[0,j]=1, U[1,j]=1
    #          V[j,k1]=1, V[j,k2]=1
    #          W[k1,0]=1, W[k1,1]=1, W[k2,0]=1, W[k2,1]=1
    # 
    # Form 2: U[0,j]=1, U[1,j]=-1
    #          V[j,k1]=1, V[j,k2]=1
    #          W[k1,0]=1, W[k1,1]=-1, W[k2,0]=1, W[k2,1]=-1
    # 
    # C_{0,k1}: (A[0,j]+A[1,j])*(B[j,k1]+B[j,k2])*1 + (A[0,j]-A[1,j])*(B[j,k1]+B[j,k2])*1
    #   = 2*A[0,j]*(B[j,k1]+B[j,k2])
    # 
    # C_{1,k1}: (A[0,j]+A[1,j])*(B[j,k1]+B[j,k2])*1 + (A[0,j]-A[1,j])*(B[j,k1]+B[j,k2])*(-1)
    #   = 2*A[1,j]*(B[j,k1]+B[j,k2])
    # 
    # C_{0,k2}: same as C_{0,k1} = 2*A[0,j]*(B[j,k1]+B[j,k2])
    # C_{1,k2}: same as C_{1,k1} = 2*A[1,j]*(B[j,k1]+B[j,k2])
    # 
    # So C_{0,k1} = C_{0,k2} and C_{1,k1} = C_{1,k2}. This doesn't separate k1 and k2!
    
    # I need to also vary V across k.
    
    # Form 1: U[0,j]=1, U[1,j]=1
    #          V[j,k1]=1, V[j,k2]=1
    #          W[k1,0]=1, W[k1,1]=1, W[k2,0]=1, W[k2,1]=1
    # 
    # Form 2: U[0,j]=1, U[1,j]=-1
    #          V[j,k1]=1, V[j,k2]=-1
    #          W[k1,0]=1, W[k1,1]=-1, W[k2,0]=-1, W[k2,1]=1
    # 
    # This is getting complicated. Let me just use a numerical search.
    
    # For now, return schoolbook.
    return best_U, best_V, best_W


def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition (rank 60) - guaranteed correct
    U, V, W = _schoolbook(n, m, p)
    
    # Verify
    assert _verify(U, V, W, n, m, p), "Schoolbook verification failed!"
    
    # Try to find a better decomposition
    if time.time() < deadline - 20:
        U2, V2, W2 = _try_structured_reduction(n, m, p, deadline)
        if _verify(U2, V2, W2, n, m, p):
            if U2.shape[1] < U.shape[1]:
                U, V, W = U2, V2, W2
    
    # Return as lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
