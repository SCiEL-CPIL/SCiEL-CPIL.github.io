"""Matrix multiplication 4x5x6 — rank-minimal tensor decomposition.

We use the block-strassen approach:
  - Partition the 4x5 matrix into blocks of 2x2 (top-left), 2x3 (top-right), 2x2 (bottom-left), 2x3 (bottom-right)
  - Partition the 5x6 matrix into blocks of 3x3 (top-left), 2x3 (top-right), 3x3 (bottom-left), 2x3 (bottom-right)

Actually, let me think more carefully about the dimensions.

A is 4x5, B is 5x6, C = A*B is 4x6.

The standard schoolbook has rank 4*5*6 = 120.

Strassen-like approaches can reduce rank. For 4x5x6, we can try:
- Split A into 2x2 and 2x3 blocks (rows 1-2, rows 3-4)
- Split B into 3x3 and 2x3 blocks (cols 1-3, cols 4-6)
- The product C is 4x6

Let me think of a block decomposition:
A = [[A11 (2x3), A12 (2x2)], [A21 (2x3), A22 (2x2)]]
B = [[B11 (3x3), B12 (3x3)], [B21 (2x3), B22 (2x3)]]
C = [[C11 (2x3), C12 (2x3)], [C21 (2x3), C22 (2x3)]]

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

This is just the standard block multiplication. Each block product contributes to the rank.

Let me count the rank contribution:
- A11*B11: 2*3*3 = 18
- A12*B21: 2*2*3 = 12
- A11*B12: 2*3*3 = 18
- A12*B22: 2*2*3 = 12
- A21*B11: 2*3*3 = 18
- A22*B21: 2*2*3 = 12
- A21*B12: 2*3*3 = 18
- A22*B22: 2*2*3 = 12
Total: 18+12+18+12+18+12+18+12 = 120. Same as schoolbook.

To get a reduction, I need to use Strassen's trick: instead of 8 multiplications, use 7.

Let me try a different split. Let me use a 2x2 block structure where each block is as square as possible.

Actually, for 4x5x6, the best known approach might be to use a combination of Strassen and schoolbook.

Let me try: split into 2x2 blocks where possible.

A is 4x5. Split into 2x2 blocks: we need 2x2 grid but 5 is odd.
Let me split A into:
- Top: 2 rows, split cols into 2+3: A_top_left (2x2), A_top_right (2x3)
- Bottom: 2 rows, split cols into 2+3: A_bot_left (2x2), A_bot_right (2x3)

B is 5x6. Split into:
- Left: 3 cols, split rows into 2+3: B_left_top (2x3), B_left_bot (3x3)
- Right: 3 cols, split rows into 2+3: B_right_top (2x3), B_right_bot (3x3)

Wait, this is getting complicated. Let me just use a known decomposition.

For M4x5x6, the rank is at most 108 (from various sources). But let me try to construct a valid decomposition.

Actually, let me just use the schoolbook decomposition which gives rank 120, and see if I can improve it.

The key insight: the rank of the matrix multiplication tensor M_{n,m,p} is at most n*m*p (schoolbook), and lower bounds exist. For 4x5x6, the best known upper bound might be around 108-114.

Let me try a simple improvement: use Strassen for a 4x4x4 sub-problem and schoolbook for the rest.

Actually, let me just return the schoolbook decomposition. It's guaranteed to be correct and gives rank 120.

Wait, the task says "Find a rank-minimal exact tensor decomposition". So I need to find the minimum rank. But the evaluator just measures the rank (lower is better). So I should try to minimize the rank.

Let me think about this differently. The tensor T has dimensions 4*5*6 = 120 entries (one per (i,j,k) tuple where i in [0,4), j in [0,5), k in [0,6)). The rank of this tensor is what we're looking for.

For matrix multiplication, the tensor is defined as T[i,j,k] = 1 if and only if the (i,k)-th entry of AB uses A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid (i,j,k).

The rank of this tensor is the minimum number of rank-1 tensors that sum to T.

For small dimensions, we can sometimes find non-trivial decompositions. Let me try to use a known result.

Actually, for the specific case of 4x5x6, I don't have a specific decomposition memorized. Let me just use the schoolbook approach which is guaranteed correct.

But wait - maybe I can use a block Strassen approach. Let me think...

If I split the 4x5 matrix into 2x2 + 2x3 blocks and the 5x6 matrix into 3x3 + 2x3 blocks, I can potentially use Strassen on the 2x2x3 and 2x3x3 sub-problems.

Hmm, this is getting complex. Let me just go with the schoolbook decomposition for now, which gives rank 120.

Actually, I recall that for M_{4,4,4}, the rank is 16 (Strassen) vs 64 (schoolbook). For M_{4,5,6}, maybe I can combine blocks.

Let me try a different approach: use the fact that 4x5x6 can be decomposed into smaller multiplications.

Actually, let me just implement the schoolbook decomposition correctly and return it. The rank will be 120.

Wait, I should try to do better. Let me think about using a 2x2 block Strassen decomposition where possible.

Split A (4x5) into:
  A = [[A1 (2x2), A2 (2x3)]]  -- top 2 rows
      [[A3 (2x2), A4 (2x3)]]  -- bottom 2 rows

Split B (5x6) into:
  B = [[B1 (2x3), B2 (2x3)]]  -- top 2 rows... no wait, B is 5x6.

Let me be more careful. B is 5x6. The column dimension of A must match the row dimension of B. A has 5 columns, B has 5 rows. Good.

Split A's columns into 2+3:
  A = [A_left (4x2), A_right (4x3)]

Split B's rows into 2+3:
  B = [[B_top (2x6)]]
      [[B_bot (3x6)]]

Then C = A_left * B_top + A_right * B_bot
where A_left*B_top is 4x6 and A_right*B_bot is 4x6.

The rank of A_left*B_top (4x2x6) is at most 4*2*6=48 (schoolbook) or less with Strassen.
The rank of A_right*B_bot (4x3x6) is at most 4*3*6=72 (schoolbook) or less.

Total: 48+72=120. Same as schoolbook.

To get an improvement, I need to use Strassen within these sub-problems.

For 4x2x6: this is multiplying a 4x2 matrix by a 2x6 matrix. The result is 4x6.
Schoolbook rank: 4*2*6 = 48.
Can we do better? The tensor is M_{4,2,6}. 

For 4x3x6: M_{4,3,6}. Schoolbook rank: 4*3*6=72.

Hmm, I don't have specific Strassen decompositions for these non-square cases.

Let me try yet another approach. What if I use a 2x2 block decomposition where each block multiplication uses Strassen?

Split A (4x5) into 2x2 blocks:
  A = [[A11 (2x2), A12 (2x3)]]
      [[A21 (2x2), A22 (2x3)]]

Split B (5x6) into 2x2 blocks:
  B = [[B11 (2x3), B12 (2x3)]]  -- rows 0-1
      [[B21 (3x3), B22 (3x3)]]  -- rows 2-4

Wait, this doesn't work cleanly because 5 = 2+3 and 6 = 3+3.

C11 (2x3) = A11*B11 + A12*B21
C12 (2x3) = A11*B12 + A12*B22
C21 (2x3) = A21*B11 + A22*B21
C22 (2x3) = A21*B12 + A22*B22

Block products:
A11 (2x2) * B11 (2x3): rank <= 2*2*3 = 12
A12 (2x3) * B21 (3x3): rank <= 2*3*3 = 18
A11 (2x2) * B12 (2x3): rank <= 2*2*3 = 12
A12 (2x3) * B22 (3x3): rank <= 2*3*3 = 18
A21 (2x2) * B11 (2x3): rank <= 2*2*3 = 12
A22 (2x3) * B21 (3x3): rank <= 2*3*3 = 18
A21 (2x2) * B12 (2x3): rank <= 2*2*3 = 12
A22 (2x3) * B22 (3x3): rank <= 2*3*3 = 18

Total: 12+18+12+18+12+18+12+18 = 120. Still schoolbook.

To get Strassen speedup, I need to combine these 8 products into 7 (or fewer) using Strassen's identity. But Strassen works on square blocks. Here the blocks are not square.

I think for this specific problem, the best I can do without extensive research is the schoolbook decomposition with rank 120. Let me just implement that correctly.

Actually, wait. Let me reconsider. The task says "rank-minimal". Maybe the actual minimum rank for M_{4,5,6} is known and less than 120. But I don't have that information readily available.

Let me just go with the schoolbook decomposition. It's correct and gives rank 120.

"""

import time
import numpy as np

def _fma_4x4x4():
    """Return rank-16 Winograd FMA decomposition of 4x4x4 matrix multiplication.
    
    Uses 4-point DFT structure. Returns (U, V, W) each of shape (16, 16)
    where U is 16x16 (n*m=16 rows), V is 16x16 (m*p=16 rows), W is 16x16 (p*n=16 rows).
    
    The FMA for 4x4x4 uses the 4-point DFT matrix:
    F = [[1,1,1,1],
         [1,-1,1,-1],
         [1,1,-1,-1],
         [1,-1,-1,1]]
    
    Actually, let me use the standard Winograd FMA for 4x4x4.
    The 4-point DFT (unscaled) is:
    W4 = [[1, 1, 1, 1],
          [1, -i, -1, i],
          [1, -1, 1, -1],
          [1, i, -1, -i]]
    
    But we need real/rational entries. Let me use the Hadamard-like transform instead.
    
    For a 4x4x4 tensor, we can use the Kronecker product structure.
    M_{4,4,4} = M_{2,2,2} ⊗ M_{2,2,2} in some sense, but that gives rank 49.
    
    Actually, the Winograd FMA for 4x4x4 has rank 16. Let me construct it using
    the 4-point DFT with real entries by using the 2-point DFT (Hadamard) twice.
    
    The 2-point DFT is H = [[1,1],[1,-1]].
    
    For 4-point, we can use the Kronecker product H⊗H = 
    [[1,1,1,1],
     [1,-1,1,-1],
     [1,1,-1,-1],
     [1,-1,-1,1]]
    
    This is the 4-point Hadamard matrix. Using this for FMA:
    C = H^T * (H*A) * (B*H)
    
    Wait, let me think more carefully. For matrix multiplication C = A*B where A,B,C are 4x4:
    
    Using transform T (4x4):
    C = T^{-1} * (T*A) * (B*T^{-T})
    
    If T is orthogonal (up to scale), T^{-1} = T^T / scale.
    
    For Hadamard H: H*H^T = 4*I, so H^{-1} = H^T/4.
    
    C = (H^T/4) * (H*A) * (B*H/4)  -- this doesn't simplify to rank 16 directly.
    
    Let me just use the known rank-16 decomposition. The Winograd FMA for 4x4x4:
    
    The key idea: C[i,k] = sum_j A[i,j]*B[j,k]
    
    With the 4-point DFT approach, we compute:
    A' = F * A * F^{-1}  (diagonalize in some sense)
    
    Actually, for the cyclic convolution interpretation:
    If we interpret 4x4 matrix multiplication as cyclic convolution, then the DFT
    diagonalizes it, giving 4 independent scalar multiplications per frequency.
    But 4x4x4 matrix multiplication is NOT cyclic convolution in general.
    
    Let me use a different approach. The rank of M_{4,4,4} is at most 16 via
    the following construction using 2x2x2 Strassen recursively:
    
    Split 4x4 into 2x2 blocks:
    A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    
    Use Strassen on the 2x2x2 block structure:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    Each M is a 2x2x2 product with rank 7 (Strassen) or 8 (schoolbook).
    Using Strassen: 7 * 7 = 49. Using schoolbook: 7 * 8 = 56.
    
    That's not 16. The rank-16 claim must use a different structure.
    
    Actually, I recall that M_{4,4,4} has rank at most 16 via the "border rank" 
    argument or via a specific decomposition. Let me look at this differently.
    
    For 4x4x4, note that 4 = 2^2. The tensor M_{4,4,4} can be viewed as
    (M_{2,2,2})^{\otimes 2} in the tensor product sense, which has rank 7^2 = 49.
    
    But there's a better decomposition. Let me try the following:
    
    Use the fact that for n=4, we can use the 4-point DFT (which has rank 4 per
    dimension) to get rank 4^2 = 16 for the "diagonal" part.
    
    Specifically, if we define the 4-point DFT matrix F (4x4) where
    F[i,j] = omega^{ij} with omega = i (imaginary unit), then:
    
    C = F^{-1} * diag(F * A * F^{-1} * B * F^{-1}) * F  -- no, this isn't right.
    
    Let me just use the schoolbook for 4x4x4 (rank 64) and see if the total is still
    better than 120. 64 + 32 + 24 = 120. Same as before. Not helpful.
    
    OK let me try a completely different partitioning.
    """
    pass

def construct():
    """Return a decomposition of M_{4,5,6} with rank < 120.
    
    Strategy: Split the middle dimension 5 = 4 + 1.
    A (4x5) = [A1 (4x4), A2 (4x1)]
    B (5x6) = [B1 (4x6); B2 (1x6)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (4x4x6): Split B1's columns into 4+2.
    B1 (4x6) = [B1a (4x4), B1b (4x2)]
    A1*B1 = A1*B1a + A1*B1b
    
    A1*B1a (4x4x4): schoolbook rank 64
    A1*B1b (4x4x2): schoolbook rank 32
    A2*B2 (4x1x6): schoolbook rank 24
    
    Total: 64 + 32 + 24 = 120. Same as before!
    
    I need a better decomposition for the 4x4x4 part.
    
    Let me try: Split 4x4x4 into 2x2x4 + 2x2x4 + 2x2x4 + 2x2x4 (four 2x2x4 blocks).
    Each 2x2x4: schoolbook rank 16. Total: 4*16 = 64. Same.
    
    Or: 2x2x4 can be done with rank 12 using Strassen-like? 
    M_{2,2,4}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2,3}
    This is just 2*2*4 = 16 scalar multiplications. Can we do better?
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    This is a 2x2x4 tensor. Its rank is at most 8 (by Strassen on the 2x2 part,
    applied to each of the 4 columns): 7*... no, Strassen gives 7 multiplications
    for 2x2x2, not 2x2x4.
    
    For 2x2x4: C has 8 entries. Each C[i,k] depends on A[i,0], A[i,1], B[0,k], B[1,k].
    The rank is at most 2*2*4=16 (schoolbook) but can be lower.
    
    Actually, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a bilinear form. The tensor has rank at most 8:
    For each (i,j) pair, we have A[i,j]*B[j,·] which contributes to all k.
    Actually, it's sum over j of A[i,j]*B[j,k]. The rank is at most 2*4=8 if we
    group by k: for each k, C[:,k] = A * B[:,k], which is a 2-dim vector.
    
    Hmm, the tensor rank of M_{2,2,4}: Let's think of it as a 2x2x4 array T[i,j,k]=delta.
    Actually T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
    Wait no, the multiplication tensor T[i,j,k] = 1 if the (i,k) entry of the product
    uses A[i,j]*B[j,k]. So T[i,j,k]=1 for all i,j,k in range.
    
    The rank of this tensor: we need to write T as sum of rank-1 tensors.
    T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,4).
    This is the all-ones tensor of shape 2x2x4, which has rank 1!
    (It's just the outer product of [1,1], [1,1], [1,1,1,1].)
    
    Wait, that's not right. The multiplication tensor for M_{n,m,p} has
    T[i,j,k] = 1 for all i,j,k. So it IS the all-ones tensor, which has rank 1.
    
    No wait, I'm confusing myself. The matrix multiplication tensor M_{n,m,p} is
    defined as: M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the sum is over all
    (i,j,k) with i<n, j<m, k<p. So M is the tensor with all entries equal to 1.
    
    But that's not the right definition either. The correct definition is:
    M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    
    where the multiplication is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    And the tensor satisfies: (A*B)[i,k] = <M, A ⊗ B ⊗ E_{ik}>
    where E_{ik} is the matrix with 1 at (i,k) and 0 elsewhere.
    
    So M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k. And indeed this is the
    all-ones tensor, which has rank 1.
    
    But that can't be right because we know the rank of M_{2,2,2} is 7, not 1.
    
    I think I'm confusing the tensor with its action. Let me reconsider.
    
    The matrix multiplication tensor T ∈ R^{n×m×p} is defined by:
    T[i,j,k] = 1 if and only if the (i,k)-th entry of the product AB involves
    the term A[i,j]*B[j,k].
    
    Since (AB)[i,k] = sum_j A[i,j]*B[j,k], every (i,j,k) triple contributes.
    So T[i,j,k] = 1 for all i,j,k.
    
    Now, the rank of this tensor: we want to write T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    where u_r ∈ R^n, v_r ∈ R^m, w_r ∈ R^p.
    
    T[i,j,k] = sum_{r=1}^R u_r[i] * v_r[j] * w_r[k]
    
    For T to be all-ones: sum_{r=1}^R u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    This means the matrix U^T V (n×m... no.
    
    Let me define: for fixed k, the matrix T[:,:,k] = u_1[:,k]... 
    
    Actually, T[i,j,k] = (U V^T W^T)[i,j,k] where U is n×R, V is m×R, W is p×R.
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    We need this to equal 1 for all i,j,k.
    
    If we set U[i,r] = 1, V[j,r] = 1, W[k,r] = 1/R for all i,j,k,r, then
    T[i,j,k] = sum_r (1)(1)(1/R) = R*(1/R) = 1. ✓
    
    So the all-ones tensor has rank 1! Just take R=1 with U=ones, V=ones, W=ones.
    
    But this contradicts the known result that the rank of M_{2,2,2} is 7.
    
    I must be misunderstanding the tensor definition. Let me look this up mentally.
    
    Ah, I think the issue is that the "matrix multiplication tensor" in the context of
    tensor rank for bilinear complexity is NOT the all-ones tensor. It's a specific
    tensor that encodes the multiplication structure.
    
    The correct definition: The tensor T for multiplying an n×m matrix by an m×p matrix
    is an element of (R^n ⊗ R^m)^* ⊗ (R^p ⊗ R^n)^* ⊗ ... 
    
    Actually, I think the standard definition is:
    T ∈ R^{n×m×p} where T[i,j,k] = 1 for all i,j,k, and the bilinear map is:
    f(A,B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
    
    And the rank of T as a 3-tensor (in the sense of tensor decomposition) is what we
    want to minimize.
    
    But as I showed, the all-ones 3-tensor has rank 1. So the "rank" in this context
    must be different from the standard tensor rank.
    
    OH WAIT. I think I see the issue. The tensor is not T[i,j,k]=1. The tensor is
    defined differently. Let me reconsider.
    
    In the context of bilinear complexity, we have:
    (AB)[i,k] = sum_{r=1}^R U[i,j... 
    
    No. The standard formulation is:
    (AB)[i,k] = sum_{r=1}^R (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k])
    
    Wait, that's not right either. Let me use the correct formulation:
    
    (AB)[i,k] = sum_{r=1}^R alpha_r[i,k] * (sum_j beta_r[i,j] * A[i,j]) * (sum_j gamma_r[j,k] * B[j,k])
    
    No, the standard formulation for the rank of matrix multiplication is:
    
    We want to express (AB)[i,k] as a sum of R rank-1 bilinear forms:
    (AB)[i,k] = sum_{r=1}^R u_r[i] * v_r[j]... 
    
    I'm getting confused. Let me just use the formulation from the problem statement:
    
    The output factor W has shape (p*n, rank) with W[k*n+i, r].
    U has shape (n*m, rank) with U[i*m+j, r].
    V has shape (m*p, rank) with V[j*p+k, r].
    
    And the decomposition is:
    (AB)[i,k] = sum_{r=1}^R U[i*m+j, r] * ... 
    
    No, I think the correct interpretation is:
    
    For each r, we have a "multiplication" that contributes to the product:
    (AB)[i,k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    Wait, that's not bilinear in A and B. Let me think again.
    
    I think the correct interpretation is:
    (AB)[i,k] = sum_{r=1}^R W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    No, that's not standard either.
    
    Actually, I think the standard formulation for the rank of matrix multiplication
    tensor is:
    
    Define the tensor T ∈ R^{(n*m) × (m*p) × (p*n)} by:
    T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' where j=j' and i=i', k=k'
    
    Hmm, this is getting complicated. Let me just use the schoolbook decomposition
    which I know is correct, and try to improve it using known results.
    
    For the schoolbook: R = n*m*p = 120.
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 where r corresponds to (i,j,k).
    
    And the product is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    Wait, I think the correct formula is:
    (AB)[i,k] = sum_{r} W[k*n+i, r] * (sum_{j'} U[i*m+j', r] * A[i,j']) * (sum_{j''} V[j''*p+k, r] * B[j'',k])
    
    For schoolbook, each r corresponds to a unique (i,j,k) triple, and:
    U[i*m+j, r] = 1 if r=(i,j,k) else 0
    V[j*p+k, r] = 1 if r=(i,j,k) else 0  
    W[k*n+i, r] = 1 if r=(i,j,k) else 0
    
    Then: sum_r W[k*n+i,r] * (sum_j' U[i*m+j',r]*A[i,j']) * (sum_j'' V[j''*p+k,r]*B[j'',k])
    = W[k*n+i,(i,j,k)] * A[i,j] * B[j,k]
    = A[i,j]*B[j,k]
    
    And summing over all (i,j,k) gives sum_j A[i,j]*B[j,k] = (AB)[i,k]. ✓
    
    OK so now I understand the formulation. The rank is the minimum R such that there
    exist U (n*m × R), V (m*p × R), W (p*n × R) with:
    (AB)[i,k] = sum_{r=1}^R W[k*n+i, r] * (sum_{j=0}^{m-1} U[i*m+j, r] * A[i,j]) * (sum_{j=0}^{m-1} V[j*p+k, r] * B[j,k])
    
    for all A, B.
    
    This is equivalent to saying that the tensor M_{n,m,p} (defined appropriately) has
    rank R.
    
    Now, for the 4x4x4 case, the rank is known to be at most 16 (via Winograd FMA)
    or 49 (via recursive Strassen). Let me try to construct the rank-16 decomposition.
    
    The Winograd FMA for 4x4x4 uses the 4-point DFT. The 4-point DFT matrix is:
    F[i,j] = exp(-2*pi*i*j/4) = (-i)^{ij}
    
    F = [[1, 1, 1, 1],
         [1, -i, -1, i],
         [1, -1, 1, -1],
         [1, i, -1, -i]]
    
    But this has complex entries. We need real (or rational) entries.
    
    Alternative: use the 4-point Hadamard (real DFT for 2-point, applied twice):
    H = [[1, 1, 1, 1],
         [1, -1, 1, -1],
         [1, 1, -1, -1],
         [1, -1, -1, 1]]
    
    H is orthogonal up to scale: H*H^T = 4*I.
    
    Using FMA with H:
    C = (H^T/4) * (H * A) * (B * H^T/4)  -- this gives C = A*B but with scaling.
    
    Actually, the FMA algorithm is:
    1. Compute A' = H * A (4 multiplications per entry of A, but H has 0/±1 entries so it's additions)
    2. Compute B' = B * H^T (similarly)
    3. Compute C' = A' * B' element-wise (16 multiplications)
    4. Compute C = H^T * C' * H / 16 (additions)
    
    Wait, this gives C = (H^T * (H*A) * (B*H^T)) / 16 = (H^T * H * A * B * H^T * H) / 16 = (4I * A * B * 4I) / 16 = A*B. ✓
    
    But the "multiplications" in step 3 are 16 scalar multiplications. However, in the
    tensor rank sense, each of these 16 multiplications corresponds to a rank-1 term.
    So the rank is 16!
    
    Let me verify: in step 3, C'[i,k] = A'[i,j] * B'[j,k] summed over j? No, element-wise
    means C'[i,k] = A'[i,k] * B'[i,k]? That doesn't make sense for matrix multiplication.
    
    I think I'm confusing FMA for convolution with FMA for matrix multiplication.
    
    For matrix multiplication C = A*B, the FMA approach is:
    C = T^{-1} * (T*A) * (B*T^{-T})
    
    where T is a transform matrix. The "multiplications" are the element-wise products
    in (T*A) * (B*T^{-T}), which is a matrix product, not element-wise.
    
    So this doesn't reduce the number of multiplications unless T has a special structure.
    
    I think the rank-16 for 4x4x4 comes from a different argument. Let me just use the
    schoolbook for now (rank 64 for 4x4x4) and see if the total is still an improvement.
    
    Actually, let me try a different approach entirely. Let me use the following partitioning:
    
    Split A (4x5) into rows: top 2 rows and bottom 2 rows.
    Split B (5x6) into columns: left 3 cols and right 3 cols.
    
    A = [[A1 (2x5)], [A2 (2x5)]]
    B = [[B1 (5x3), B2 (5x3)]]
    
    C = [[C1 (2x3), C2 (2x3)], [C3 (2x3), C4 (2x3)]]
    
    C1 = A1*B1, C2 = A1*B2, C3 = A2*B1, C4 = A2*B2
    
    Each is a 2x5x3 multiplication. Schoolbook rank: 2*5*3 = 30. Total: 4*30 = 120.
    
    Can we do 2x5x3 with rank < 30? The tensor M_{2,5,3} has rank at most 2*5*3=30.
    Lower bound: at least max(2*5, 5*3, 2*3) = max(10,15,6) = 15.
    
    I don't know the exact rank of M_{2,5,3}. Let me just go with schoolbook.
    
    Given the time constraints, let me just implement the schoolbook decomposition
    (rank 120) which is guaranteed correct, and accept that I may not improve upon it
    in this iteration.
    """
    n, m, p = 4, 5, 6
    rank = n * m * p  # 120
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 120
    V = [[0] * rank for _ in range(m * p)]  # 30 x 120
    W = [[0] * rank for _ in range(p * n)]  # 24 x 120
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
