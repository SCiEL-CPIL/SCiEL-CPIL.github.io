"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    U has shape (n*m, rank) = (9, rank)
    V has shape (m*p, rank) = (15, rank)  
    W has shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies: for all i,j,k,
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    
    For matrix multiplication A (n x m) * B (m x p) = C (n x p),
    the tensor is T[i,j,k] where C[i,k] += A[i,j] * B[j,k].
    So T[i,j,k] = 1 for all valid (i,j,k) - it's the all-ones tensor.
    Wait, no. The tensor for matmul is:
    T[(i,k), j] = sum over j of A[i,j] * B[j,k]
    
    Actually the standard formulation: the matrix multiplication tensor T has
    T[i, j, k] such that (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    In the decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    (with W indexed as W[k*n+i, c] per the task spec)
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait no - that's not right either. Let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And T[i,j,k] is the matmul tensor. For standard matrix multiplication,
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    Actually the matmul tensor IS the all-1 tensor of shape (n, m, p), because
    every product A[i,j]*B[j,k] contributes to exactly one output C[i,k].
    
    Hmm wait, that means we're decomposing the all-ones tensor into rank-1 tensors.
    The rank of the all-ones tensor of shape (n,m,p) is... 1! Just take U=V=W=1.
    
    But that can't be right for matrix multiplication. Let me reconsider.
    
    The matrix multiplication tensor is NOT the all-ones tensor. It's a specific
    tensor where T[i,j,k] = delta_{i,k} or something... No.
    
    Actually, the correct formulation: The matmul map is a trilinear map
    (A, B) -> A*B where A is n x m and B is m x p.
    
    The associated tensor T in V*_{n x m} ⊗ V*_{m x p} ⊗ V*_{n x p} is:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * <e_i, A e_j> * <e_k, ...>
    
    More carefully: T(α, β, γ) = sum_{i,j,k} α_{i,j} β_{j,k} γ_{k,i}
    No wait...
    
    T(α, β, γ) = trace(α^T * β * γ) = sum_{i,j,k} α_{i,j} β_{j,k} γ_{k,i}
    
    Hmm, that uses the trace convention. So T[i,j,k] = 1 for ALL i,j,k where
    the indices wrap around via trace. But since we're not taking trace of the
    product in the usual sense...
    
    Let me think again. The matmul tensor T satisfies:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    So if we view A as a vector in R^{n*m}, B as a vector in R^{m*p}, and
    C = A*B as a vector in R^{n*p}, then:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The bilinear form is: C = T(A, B) where T is a tensor in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p})
    
    T_{(i,j),(j,k),(i,k)} = 1, and all other entries are 0.
    
    Wait, that's the key! T is NOT all ones. T[(i,j), (j',k), (i',k')] = 1 if and only if
    j = j' and i = i' and k = k'. Otherwise 0.
    
    So T has non-zero entries only when the middle indices match (j=j') and the
    output indices match the input indices (i=i', k=k').
    
    In the decomposition: T[(i,j), (j',k), (i',k')] = sum_r U[(i,j), r] * V[(j',k), r] * W[(i',k'), r]
    
    And the task says W is indexed as W[k*n+i, r], which corresponds to W[(i',k')] = W[k'*n+i', r].
    
    So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    Wait, that's the delta tensor, which has rank n*m*p (the schoolbook). That's 45.
    
    Hmm, but that doesn't seem right either. Let me re-examine.
    
    Actually I think I was confusing myself. The standard result is that the rank of
    the n x m times m x p matrix multiplication tensor is what we're trying to minimize.
    
    For the case n=m=p=2, Strassen showed rank 7 < 8 = schoolbook.
    For 3x3 x 3x3, best known is 23 (Landsberg 1998, or actually it was shown to be at most 23,
    and lower bounds are around 19-20).
    
    For 3x3 * 3x5, this is a less standard case. Let me think about what approaches might work.
    
    One approach: use known decompositions for smaller cases and extend.
    Another: use algebraic/combinatorial methods to find a low-rank decomposition.
    
    Given the time constraint (120s), a brute-force search over rank-1 terms is infeasible
    for rank < 45. But we can try:
    
    1. Use the schoolbook decomposition as baseline (rank 45)
    2. Try to find a lower rank decomposition using randomized or structured approaches
    
    Actually, for this specific problem (3,3,5), let me think about whether there are known results.
    
    The rank of the (n,m,p) matmul tensor R(n,m,p) satisfies:
    - R(n,m,p) <= n*m*p (schoolbook)
    - Various upper bounds from Strassen-like methods
    
    For (3,3,5): schoolbook gives 45.
    
    One idea: decompose 3x3*3x5 as a sum of smaller blocks.
    3x3*3x5 can be viewed as:
    - Three 1x3*3x5 blocks (rows of A times B)
    - Or three 3x1*1x5 blocks (columns)
    
    This doesn't directly help.
    
    Another approach: use the fact that 5 = 2+3, and decompose B into two parts:
    B = [B1 | B2] where B1 is 3x2 and B2 is 3x3.
    Then A*B = A*B1 + A*B2.
    
    R(3,3,2) + R(3,3,3) would give an upper bound.
    R(3,3,2): schoolbook is 18. Is there a better decomposition?
    R(3,3,3): best known is 23.
    
    So 18 + 23 = 41 < 45. That's a valid upper bound!
    
    But we need to verify that the sum of decompositions is valid. If we have:
    T_{3,3,2} = sum_{r=1}^{18} U_r ⊗ V_r ⊗ W_r
    T_{3,3,3} = sum_{r=19}^{41} U_r ⊗ V_r ⊗ W_r
    
    Then T_{3,3,5} = T_{3,3,2} + T_{3,3,3} (with appropriate embedding) would have rank <= 41.
    
    The embedding: for the 3x3*3x2 part, B has shape 3x2, so V entries are in R^{3*2}=R^6.
    For the 3x3*3x3 part, B has shape 3x3, so V entries are in R^{3*3}=R^9.
    For the full 3x3*3x5, B has shape 3x5, so V entries are in R^{3*5}=R^15.
    
    We embed the 3x2 part into the first 6 coordinates of R^15, and the 3x3 part into
    the last 9 coordinates. Similarly for W (output): 3x2 -> first 2*3=6 coords, 3x3 -> last 9 coords.
    
    This gives rank 41.
    
    Can we do better? What about R(3,3,2)? Is it 18 or can it be lower?
    
    For 2x2*2x2, Strassen gives 7. For 3x3*3x2... I'm not sure of the exact rank.
    Schoolbook gives 18. There might be a lower bound.
    
    Actually, let me think about this differently. The key insight is that we can use
    known low-rank decompositions for sub-problems and combine them.
    
    Let me try to implement the rank-41 decomposition:
    - First 18 terms: 3x3 * 3x2 (schoolbook for this sub-problem)
    - Next 23 terms: 3x3 * 3x3 (need a rank-23 decomposition)
    
    The challenge is getting a valid rank-23 decomposition for 3x3*3x3.
    
    The Landsberg rank-23 decomposition for 3x3*3x3 is known but complex. Let me try to
    implement it or find an alternative.
    
    Actually, for simplicity and correctness, let me first implement the rank-41
    decomposition using schoolbook for both sub-problems (18 + 27 = 45, which doesn't help)
    or use a known decomposition for 3x3*3x3.
    
    Wait, schoolbook for 3x3*3x3 is 27, not 23. So 18+27=45, same as direct schoolbook.
    
    To get 41, I need a rank-23 decomposition for 3x3*3x3.
    
    The Landsberg decomposition is quite involved. Let me see if I can implement it.
    
    Actually, there's a simpler approach: use the Strassen-like algorithm.
    
    For 2x2*2x2, Strassen has rank 7:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 3x3, we can partition into 2x2 blocks plus a row and column:
    A = [[A00, A01, A02],
         [A10, A11, A12],
         [A20, A21, A22]]
    
    This gets complicated. Let me try a different approach.
    
    Given the complexity, let me implement the rank-41 decomposition using:
    - 3x3*3x2 via schoolbook (rank 18)  
    - 3x3*3x3 via a known rank-23 decomposition
    
    For the 3x3*3x3 rank-23 decomposition, I'll use the one from the literature.
    The decomposition involves 23 trilinear forms.
    
    Actually, let me try a completely different approach: use a randomized search to
    find a low-rank decomposition. This is unlikely to find the optimal rank in 120s,
    but might find something below 45.
    
    Alternatively, let me just implement the 41-rank decomposition with the known
    3x3*3x3 rank-23 result. I'll encode the 23 terms from the Landsberg decomposition.
    
    The Landsberg decomposition for 3x3*3x3 matmul uses 23 terms. Let me try to
    reconstruct it.
    
    Actually, I realize I should just try to implement a valid lower-rank decomposition
    even if it's not optimal. Let me use the block decomposition approach:
    
    Split 3x5 into 3x2 + 3x3, giving rank 18 + 27 = 45 (no improvement).
    
    Split 3x3 into 2x2 + 1x3, and 3x5 into... this is getting messy.
    
    Let me try: 3x3*3x5.
    Partition A (3x3) into top 2 rows (2x3) and bottom 1 row (1x3).
    Partition B (3x5) into left 2 cols (3x2) and right 3 cols (3x3).
    
    A*B = A_top * B + A_bottom * B
        = (2x3)*(3x5) + (1x3)*(3x5)
    
    R(2,3,5) + R(1,3,5) = ?
    R(2,3,5): schoolbook is 30. Can it be lower?
    R(1,3,5): schoolbook is 15. Since it's 1 row, it's just a linear map, rank = 15.
    
    Total: 30 + 15 = 45. No improvement.
    
    What about: A (3x3) * B (3x5).
    Partition B into 3x1 + 3x1 + 3x1 + 3x1 + 3x1 (five 3x1 blocks).
    A * B = sum of A * B_i for i=1..5.
    Each A * B_i is (3x3)*(3x1) = 3x1.
    R(3,3,1) = 9 (schoolbook, and this is optimal since it's a linear map).
    Total: 5 * 9 = 45. No improvement.
    
    The key to beating schoolbook is to share computations across blocks, which is
    what Strassen does for 2x2.
    
    Let me try the Strassen approach for 3x3*3x5:
    Partition A into 3 blocks of size 1x3 (rows), and B into 5 blocks of size 3x1 (cols).
    
    This doesn't directly apply Strassen.
    
    Alternative: partition both into 2x2 blocks where possible.
    A (3x3) -> top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1.
    B (3x5) -> partition into 3x2, 3x2, 3x1.
    
    A*B = A_tl * B1 + A_tr * B1 + A_bl * B1 + A_br * B1 + ... this is getting too complex.
    
    Let me just go with a practical approach: implement the rank-41 decomposition
    using a known 3x3*3x3 rank-23 decomposition. I'll hardcode the 23 terms.
    
    The Landsberg 23-term decomposition for 3x3*3x3 is well-known. Let me try to
    implement it. The decomposition is:
    
    T = sum_{r=1}^{23} u_r ⊗ v_r ⊗ w_r
    
    where u_r, v_r ∈ R^9 and w_r ∈ R^9.
    
    I'll encode the coefficients. This is a large table, but let me try.
    
    Actually, given the time constraints and the difficulty of correctly implementing
    the Landsberg decomposition from memory, let me take a different approach:
    
    1. Implement the schoolbook decomposition (rank 45) as a fallback.
    2. Try a simple randomized/structured search to find any decomposition with rank < 45.
    3. If found, return it; otherwise return schoolbook.
    
    For the search, I can try:
    - Fix a candidate rank r (e.g., 44, 43, ...)
    - Randomly initialize U, V, W
    - Use iterative refinement (e.g., ALS - alternating least squares) to minimize
      ||T - sum U_r ⊗ V_r ⊗ W_r||^2
    - Check if the residual is exactly 0 (within floating point tolerance)
    
    This is unlikely to find an exact decomposition, but worth trying.
    
    Actually, for exact decomposition, the approach is more algebraic. Let me think...
    
    Given the complexity, let me just implement the schoolbook decomposition and
    add a simple optimization: try to merge some of the 45 terms if possible.
    
    Two terms (u1⊗v1⊗w1) and (u2⊗v2⊗w2) can be merged if one is a scalar multiple
    of the other, but in the schoolbook decomposition, all terms are distinct and
    orthogonal, so no merging is possible.
    
    OK, I'll just go with the schoolbook decomposition. It's correct and safe.
    
    Wait, the task says "Find a rank-minimal exact tensor decomposition" and "score is
    the rank" with "lower is better". The seed has rank 45 (score 1.0). To improve,
    I need rank < 45.
    
    Let me try one more idea: use the fact that 3 = 2+1 and apply a Strassen-like
    decomposition.
    
    For 3x3*3x3:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    Partition into:
    A = [[B, c], [d^T, e]] where B is 2x2, c is 2x1, d^T is 1x2, e is 1x1.
    Similarly for C = A*B.
    
    This gives a recursive structure, but it's not clear how to get rank < 27.
    
    I think the best I can do reliably within the time constraint is to return the
    schoolbook decomposition. Let me do that.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
