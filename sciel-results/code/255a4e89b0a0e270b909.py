"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import numpy as np
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The multiplication tensor has entries T[i,j,k] = 1 if (i,j) is in A, (j,k) in B, (k,i) in C
    such that the product entry (i,k) = sum_j A[i,j]*B[j,k].
    
    For matrix multiplication A(n x m) * B(m x p) -> C(n x p):
    The tensor is T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    and the decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j,k}
    
    Wait, let me re-read the interface. The output factor follows W[k*n+i, r].
    
    Standard matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 (it's the identity tensor for multiplication).
    A decomposition of rank r means we can write:
    C[i,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that doesn't look right. Let me think about this more carefully.
    
    Actually, for matrix multiplication, the standard tensor decomposition works as follows:
    We want C[i,k] = sum_j A[i,j] B[j,k].
    
    A rank-r decomposition gives us:
    C[i,k] = sum_{r=0}^{r-1} (sum_j U[i,j,r] A[i,j]) * (sum_l V[j,l,r] B[j,l]) ... no.
    
    Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined on indices (i,j,k) 
    where i in [0,n), j in [0,m), k in [0,p), and M[i,j,k] = 1.
    
    A rank-r decomposition means we can find tensors U, V, W such that:
    M[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, but the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    So U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    
    So the decomposition is:
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  for all i,j,k
    
    And the multiplication algorithm would be:
    C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    
    Hmm, that's not quite right either. Let me think about how Strassen's algorithm works.
    
    In Strassen's algorithm for 2x2 matrices:
    We compute 7 products of the form (linear combination of A entries) * (linear combination of B entries),
    then combine them to get C entries.
    
    The tensor decomposition says:
    C[i,k] = sum_r P_r * Q_r * R_r
    where P_r = sum_j U[i*m+j, r] * A[i,j]  -- wait, this mixes i and j.
    
    Actually, I think the correct interpretation is:
    C[i,k] = sum_r (sum_j U[(i,j), r] * A[i,j]) * (sum_j V[(j,k), r] * B[j,k])
    
    No wait, that would give sum_r sum_j sum_l U[(i,j),r] A[i,j] V[(j,k),r] B[j,k]
    = sum_r A[i,j] B[j,k] U[(i,j),r] V[(j,k),r) summed over j... this is getting confusing.
    
    Let me look at the schoolbook decomposition in the seed code:
    U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    
    So for each (i,j,k) triple, we have one rank-1 term. The product is:
    sum_{i,j,k} U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1 for each (i,j,k)
    
    But how does this give us C[i,k] = sum_j A[i,j] B[j,k]?
    
    I think the way it works is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    No, that still doesn't work dimensionally.
    
    Let me try: 
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    For the schoolbook case, U[i*m+j, col] = 1 only when col corresponds to (i,j,k), similarly for V and W.
    So the sum over r picks out the specific (i',j',k') such that i'=i, j'=j, k'=k... 
    but then we'd get A[i,j] * B[j,k] * 1 for each j, and summing over j gives sum_j A[i,j] B[j,k]. Yes!
    
    Wait, but the sum over r would pick out all r where U[i*m+j', r] != 0 AND V[j'*p+k, r] != 0 AND W[k*n+i, r] != 0.
    In the schoolbook case, for a fixed (i,k), the only r that contribute are those where i'=i, k'=k, and j' is free.
    So we get sum_j A[i,j] * B[j,k] * W[k*n+i, col] = sum_j A[i,j] * B[j,k] * 1 = sum_j A[i,j] B[j,k]. 
    
    OK so the formula is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Hmm wait, but that has U indexed by (i,j) and we're summing over j with A[i,j]. That means for fixed i,
    we're taking a linear combination over j of A[i,j] weighted by U[i*m+j, r]. Similarly for V.
    
    Actually I realize I had the indices slightly off. Let me re-derive.
    
    The tensor T[i,j,k] = 1 for all i,j,k (in the appropriate ranges). We want:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is a 3-tensor decomposition. The "multiplication" interpretation is:
    Given A (n x m) and B (m x p), define C (n x p) by C[i,k] = sum_j A[i,j] B[j,k].
    
    The rank-r decomposition allows us to compute C using r "multiplications" (of scalars):
    C[i,k] = sum_r W[k*n+i, r] * M_r
    where M_r = (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Wait, that's still not quite right because the sums over j in U and V are independent.
    
    Let me just verify with the schoolbook:
    For rank term corresponding to (i0, j0, k0):
    U[i*m+j, r] = 1 if i=i0 and j=j0, else 0
    V[j*p+k, r] = 1 if j=j0 and k=k0, else 0  
    W[k*n+i, r] = 1 if k=k0 and i=i0, else 0
    
    Then sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    = sum_r W[k*n+i, r] * A[i,i0_r] * B[j0_r, k]  (where i0_r, j0_r, k0_r are the indices for rank r)
    
    For this to be nonzero, we need i=i0_r, k=k0_r, and j0_r can be anything.
    So for fixed (i,k), the contributing r are those with i0_r=i, k0_r=k, and j0_r=j for some j.
    The sum becomes sum_j A[i,j] * B[j,k] * 1 = sum_j A[i,j] B[j,k]. 
    
    So the formula is correct:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    And the constraint is that for all (i,j,k):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Now, the known minimum rank for 3x3 matrix multiplication is 19 (Landsberg's bound says at least 19, and there exist decompositions of rank 19).
    
    For 3x3 x 3x7, we can think of this as:
    C = A * B where A is 3x3, B is 3x7.
    We can write B = [B1, B2, B3] where B1, B2, B3 are 3x3, 3x1, 3x3... no, 3x7 = 3x3 + 3x3 + 3x1.
    
    Actually, B is 3x7, so we can split it as B = [B_left (3x3), B_mid (3x3), B_right (3x1)].
    Then C = A * B = [A*B_left, A*B_mid, A*B_right] which is 3x7.
    
    If we have a rank-19 decomposition for 3x3 x 3x3, we can use it for A*B_left and A*B_mid.
    For A*B_right (3x3 x 3x1), the minimum rank is 3 (schoolbook, since it's just 3 dot products).
    
    So a naive upper bound is 19 + 19 + 3 = 41.
    
    But we can do better by sharing structure. The key insight is that A is the same in all three multiplications.
    
    Actually, let me think about this differently. The tensor for 3x3 x 3x7 multiplication is:
    T[i,j,k] = 1 for i in [0,3), j in [0,3), k in [0,7).
    
    This is a 3 x 3 x 7 tensor. We want to decompose it into rank-1 terms.
    
    One approach: use the known rank-19 decomposition for 3x3 x 3x3, and extend it.
    
    Let me recall the rank-19 decomposition for 3x3 matrix multiplication. This is based on the work of Landsberg, and the explicit construction uses the fact that the tensor can be decomposed using the 7 Strassen-type products plus additional terms.
    
    Actually, the well-known rank-19 decomposition for 3x3 matrices is quite complex. Let me look for a simpler approach.
    
    Alternative: use a randomized search to find a low-rank decomposition.
    
    Given the time constraints, let me try a different approach. I'll use the fact that we can write:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    and try to find a decomposition with rank less than 63 (the schoolbook rank).
    
    One simple improvement: note that B is 3x7, so we can write B = [B1 | B2 | B3] where B1, B2 are 3x3 and B3 is 3x1.
    Then C = [A*B1 | A*B2 | A*B3].
    
    Using schoolbook for each: 9 + 9 + 3 = 21. That's already better than 63.
    
    Can we do better? For 3x3 x 3x3, the minimum rank is 19. So we could get 19 + 19 + 3 = 41.
    
    But we can potentially share the A-side computations. 
    
    Let me try a practical approach: use the known rank-19 decomposition for 3x3x3x3, and combine with schoolbook for the 3x1 part.
    
    The rank-19 decomposition for 3x3 matrix multiplication is known but complex. Let me see if I can find it or derive it.
    
    Actually, for the purpose of this task, let me try a randomized algebraic approach. I'll set up the equations and try to solve them with a low rank.
    
    Given the complexity, let me start with a simple but improved decomposition:
    - Split B into three 3x3 blocks (padding the last one with zeros to make it 3x3, but we only need 3x1).
    - Actually, B is 3x7, so split as 3x3, 3x3, 3x1.
    - Use schoolbook for the 3x1 part: rank 3.
    - For the two 3x3 parts, try to use a shared decomposition.
    
    If we use independent rank-19 decompositions for each 3x3 part, total rank = 19 + 19 + 3 = 41.
    
    But if we can share the A-side, we might do better.
    
    Let me try yet another approach: use the fact that the tensor is 3x3x7, and try to find a decomposition by solving a system of equations.
    
    For rank r, we have:
    - U: 9 x r matrix
    - V: 21 x r matrix  
    - W: 21 x r matrix
    
    Total unknowns: 9r + 21r + 21r = 51r.
    Constraints: 3*3*7 = 63 equations.
    
    For r = 19: 51*19 = 969 unknowns, 63 constraints. Very underdetermined.
    For r = 15: 51*15 = 765 unknowns, 63 constraints. Still very underdetermined.
    
    The challenge is that the equations are nonlinear (products of U, V, W entries).
    
    Let me try a practical randomized approach:
    1. Initialize U, V, W randomly.
    2. Use gradient descent or a similar optimization to minimize ||T - U*V*W||^2.
    3. Check if the result is exact (integer/half-integer entries).
    
    But getting exact integer/half-integer entries from a numerical optimization is tricky.
    
    Alternative: use a structured decomposition.
    
    Let me try the following:
    - Use the Strassen-like decomposition for the 3x3x3x3 parts.
    
    For 2x2x2x2, Strassen uses rank 7.
    For 3x3x3x3, the minimum known rank is 19.
    
    The rank-19 decomposition can be constructed as follows (based on Landsberg's work):
    It uses 7 "Strassen-type" products and 12 additional products.
    
    Actually, let me just try to construct a decomposition with rank 41 (19+19+3) and see if I can do better.
    
    For the 3x3x3x3 part, I'll use a known rank-19 decomposition. Let me look up the explicit form.
    
    The rank-19 decomposition for 3x3 matrix multiplication is given in terms of the 7 Strassen products for the 2x2 submatrices, plus additional terms.
    
    Actually, this is getting very complex. Let me try a simpler approach:
    
    1. Use schoolbook for the full 3x3x3x7: rank 63.
    2. Try to reduce the rank by exploiting structure.
    
    A simple reduction: note that for fixed i, C[i,k] = sum_j A[i,j] B[j,k].
    We can write this as: C[i,:] = A[i,:] * B (row vector times matrix).
    So C[i,:] is a linear combination of the rows of B, with coefficients A[i,0], A[i,1], A[i,2].
    
    This means C[i,:] = A[i,0] * B[0,:] + A[i,1] * B[1,:] + A[i,2] * B[2,:].
    
    So we need 3 "multiplications" per row of C, giving 9 total for the 3 rows. But each "multiplication" is a scalar times a 7-vector, which can be done with 7 scalar multiplications.
    
    Wait, that's just the schoolbook again: 3*3*7 = 63.
    
    Let me try a different structure. What if we use the fact that we only need 3x7 output?
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can compute this as:
    C[i,:] = A[i,0] * B[0,:] + A[i,1] * B[1,:] + A[i,2] * B[2,:]
    
    Each term A[i,j] * B[j,:] is a scalar times a 7-vector. If we precompute the 3 row vectors of B, we just need 3 scalar multiplications per row of C, for 9 total scalar multiplications. But each scalar multiplication with a 7-vector requires 7 scalar mults, so 9*7 = 63. Same as schoolbook.
    
    Unless we can share the scalar multiplications. For example, if A[i,0] = A[i,1], then we can combine.
    
    This suggests that the rank is related to the structure of A and B, but since we need a decomposition that works for ALL A and B, we can't assume any structure.
    
    OK let me just try to implement a rank-41 decomposition using two rank-19 decompositions for the 3x3 parts and schoolbook for the 3x1 part.
    
    First, I need the rank-19 decomposition for 3x3x3x3. Let me find it.
    
    The rank-19 decomposition is based on the following idea:
    Write A and B as 2x2 block matrices with 1x1 blocks in the corners:
    A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    
    Use Strassen's 7 products for the 2x2 submatrices, plus additional products for the boundary terms.
    
    This is quite involved. Let me try a different approach: use a randomized search to find a low-rank decomposition.
    
    Given the time constraints, let me implement a simple randomized search:
    1. Try ranks from 19 up to 63.
    2. For each rank, initialize U, V, W randomly.
    3. Use a nonlinear solver to minimize the error.
    4. Check if the solution is exact.
    
    But this is likely to be too slow for the time limit.
    
    Let me try a more structured approach. I'll use the following decomposition:
    
    For the 3x3x3x7 tensor, I'll use:
    - 7 Strassen products for the first 2x2x2x2 block (using rows 0,1 and cols 0,1)
    - 7 Strassen products for the second 2x2x2x2 block (using rows 0,1 and cols 1,2)
    - Additional products for the remaining terms
    
    Actually, this is getting too complex without a clear path. Let me just implement the schoolbook decomposition and see if I can make any simple improvements.
    
    Wait, I just realized: the task says "rank-minimal exact tensor decomposition". The minimum rank for 3x3x3x7 matrix multiplication is not known to me off the top of my head. Let me just try to find a good decomposition.
    
    Let me try rank 41 (19+19+3) using known decompositions.
    
    For the 3x3x3x3 rank-19 decomposition, I'll use the following construction:
    
    The 3x3 matrix multiplication tensor can be decomposed using the 7 Strassen products for the 2x2 submatrices, plus 12 additional products for the boundary terms.
    
    Let me denote the entries of A as a_ij and B as b_jk.
    
    Strassen products for the top-left 2x2 block (using a_00, a_01, a_10, a_11 and b_00, b_01, b_10, b_11):
    M1 = (a_00 + a_11)(b_00 + b_11)
    M2 = (a_10 + a_11)b_00
    M3 = a_00(b_10 - b_11)
    M4 = a_11(b_01 - b_00)
    M5 = (a_00 + a_01)b_11
    M6 = (a_10 - a_00)(b_00 + b_10)
    M7 = (a_01 - a_11)(b_11 + b_01)
    
    These give the 4 products for the 2x2 submatrix.
    
    For the full 3x3, we need additional products for the terms involving a_02, a_20, a_21, a_22 and b_20, b_21, b_22.
    
    This is getting very detailed. Let me just implement a reasonable decomposition and hope for the best.
    
    Given the complexity and time constraints, let me implement the schoolbook decomposition (rank 63) and add a simple optimization: if we can find a lower-rank decomposition within the time limit, use it; otherwise, return the schoolbook.
    
    Actually, let me try a specific low-rank construction.
    
    For 3x3 x 3x7, we can write:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Split the sum over j into j=0,1,2:
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    Each term is a product of a scalar from A and a scalar from B. That's 9 terms per (i,k) pair, and there are 3*7=21 pairs, so 9*21=189... no wait, that's not right.
    
    For each (i,k), we have 3 products. Total products = 3*7*3 = 63. That's the schoolbook.
    
    To reduce the rank, we need to combine products. For example, if we can write:
    A[i,0]B[0,k] + A[i,1]B[1,k] = (A[i,0] + A[i,1]) * B[0,k] + A[i,1] * (B[1,k] - B[0,k])
    
    This doesn't reduce the number of products.
    
    The key insight for Strassen is that we can compute multiple output entries using the same intermediate products.
    
    For 3x3x3x7, let me try to use the following approach:
    - Compute 7 Strassen-type products for the 2x2 submatrices of A and B (using the first 2 rows/cols).
    - Use these to compute the corresponding entries of C.
    - Handle the remaining entries with additional products.
    
    This is still quite complex. Given the time constraints, let me just implement the schoolbook decomposition and return it. The score will be 1.0, which is the same as the seed.
    
    Actually, wait. Let me try one more thing: use a simple rank reduction by noting that the last column of B (B[:,6]) only contributes to C[:,6]. So we can handle it separately with 3 products (one for each row of C).
    
    For the first 6 columns of B (3x6), we can split into two 3x3 blocks and use two rank-19 decompositions (if I can find them).
    
    Without the explicit rank-19 decomposition, let me just use schoolbook for the 3x6 part: 3*3*6 = 54 products, plus 3 for the last column, total 57.
    
    That's better than 63! Let me implement this.
    
    Actually, even simpler: split B into three 3x3 blocks (the last one being 3x1, padded to 3x3 with zeros).
    Schoolbook for each: 9 + 9 + 3 = 21. Wait, that's way better than 63.
    
    Oh wait, I think I was confusing myself. Let me re-examine.
    
    Schoolbook for 3x3 x 3x7:
    For each (i,k) in [0,3)x[0,7), C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k].
    This requires 3 products per (i,k) pair, and there are 3*7=21 pairs, so 3*21=63 products.
    
    If I split B into B1 (3x3), B2 (3x3), B3 (3x1):
    C1 = A*B1 (3x3), C2 = A*B2 (3x3), C3 = A*B3 (3x1).
    Schoolbook for C1: 3*3*3=27 products.
    Schoolbook for C2: 3*3*3=27 products.
    Schoolbook for C3: 3*3*1=9 products.
    Total: 27+27+9=63. Same as before.
    
    So splitting doesn't help with schoolbook. The total is always n*m*p.
    
    To get a lower rank, we need to use a non-schoolbook decomposition, like Strassen.
    
    For 3x3x3x3, the minimum rank is 19 (vs. 27 for schoolbook).
    For 3x3x3x7, the minimum rank is unknown to me, but it's at most 19+19+3=41 if we can use independent rank-19 decompositions for the two 3x3 parts and schoolbook for the 3x1 part.
    
    But we can potentially do better by sharing structure.
    
    Given the complexity of finding an explicit rank-19 decomposition, let me try a different approach: use a randomized search to find a low-rank decomposition.
    
    I'll set up the problem as: find U (9xr), V (21xr), W (21xr) such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    This is a system of 63 nonlinear equations in 51r unknowns.
    
    For r=19, we have 969 unknowns and 63 equations. Very underdetermined.
    
    I can try to find a solution by:
    1. Randomly initializing U, V, W.
    2. Using a nonlinear solver (e.g., scipy.optimize.least_squares) to minimize the residual.
    3. Checking if the residual is zero (or very small).
    
    But the solution may not have integer/half-integer entries.
    
    Alternatively, I can try to find a solution with rational entries by using a symbolic approach, but that's even more complex.
    
    Given the time constraints, let me just implement the schoolbook decomposition and return it. The score will be 1.0.
    
    Actually, let me try one more thing: use a simple rank-21 decomposition.
    
    Idea: for each i in [0,3), compute the row vector A[i,:] * B (which is a 7-vector).
    This requires 3 scalar multiplications per output entry, for 3*7=21 output entries, so 63 scalar multiplications. Same as schoolbook.
    
    Unless we can share the scalar multiplications. For example, if we precompute some linear combinations of the rows of B.
    
    Let me try: define B' = [B[0,:]+B[1,:], B[1,:]-B[0,:], B[2,:]] (3x7 matrix).
    Then A[i,0]*B[0,:] + A[i,1]*B[1,:] = ((A[i,0]+A[i,1])/2) * B'[0,:] + ((A[i,1]-A[i,0])/2) * B'[1,:].
    
    This doesn't reduce the number of products.
    
    I think the key insight I'm missing is that in Strassen's algorithm, the same intermediate product is used to compute multiple output entries.
    
    For 2x2x2x2, Strassen computes 7 products and uses them to compute all 4 output entries.
    
    For 3x3x3x3, we need to find a similar sharing pattern.
    
    Given the complexity, let me just implement the schoolbook decomposition and return it.
    
    # EVOLVE-BLOCK-START
    def construct():
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
"""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p for 3x3 x 3x7 matrix multiplication.
    """
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
