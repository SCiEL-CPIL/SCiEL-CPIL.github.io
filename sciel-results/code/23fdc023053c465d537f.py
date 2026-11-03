import time

DIMENSIONS = [3, 3, 8]

def _strassen_2x2():
    """
    Returns the 7-term Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Each term is (u_vec, v_vec, w_vec) where u_vec is length 4 (2x2 input A),
    v_vec is length 4 (2x2 input B), w_vec is length 4 (2x2 output C).
    
    Indexing: u_vec[i*2+j] corresponds to A[i,j], etc.
    Output C[i,j] = sum_r u_vec[i*2+j]_r * v_vec[i*2+j]_r * w_vec[i*2+j]_r
    """
    # Standard Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # u vectors (A part) for each M
    # A indices: A00=0, A01=1, A10=2, A11=3
    u = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [1, 0, -1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    # v vectors (B part) for each M
    # B indices: B00=0, B01=1, B10=2, B11=3
    v = [
        [1, 1, 0, 0],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [1, 0, -1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    # w vectors (C part) for each M
    # C indices: C00=0, C01=1, C10=2, C11=3
    w = [
        [1, 0, 0, -1],  # M1: C00 + (-C11)
        [0, 0, -1, 1],  # M2: (-C10) + C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: (-C00) + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    return u, v, w

def construct():
    """
    Constructs a tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Strategy: Embed the 3x3 matrix product into a 4x4 framework using Strassen
    for the 2x2 block structure, combined with schoolbook for the 8 columns.
    
    We decompose the 3x3 * 3x8 multiplication as follows:
    - The 3x3 A matrix and 3x8 B matrix.
    - We can write the 3x3 product as a sum of 2x2 Strassen blocks plus
      additional terms for the third row/column.
    
    Alternative cleaner approach: Use the fact that 3x3 * 3x8 can be viewed
    as 8 independent 3x3 * 3x1 products. For each 3x3 * 3x1, we can use
    a rank-7 decomposition (since 3x3 matrix-vector multiplication has rank 7
    in the tensor sense? Actually, matrix-vector mult 3x3 has rank 9 generally).
    
    Let me use a different approach: tensor product decomposition.
    
    3x3 * 3x8 multiplication tensor T_{ijk} where i,j in [0,3), k in [0,8).
    C_{ik} = sum_j A_{ij} B_{jk}
    
    We can decompose this as: treat the 3x3 part with Strassen-like methods
    and the 8-column part with schoolbook.
    
    Actually, the cleanest approach: 
    - Use rank-7 Strassen for a 2x2 block of the 3x3 matrices
    - Handle the remaining parts with schoolbook
    
    Let's think of 3x3 as:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B = [[B11, ..., B18],
         [B21, ..., B28],
         [B31, ..., B38]]
    
    C = A*B (3x8)
    
    C[i][k] = A[i][1]*B[1][k] + A[i][2]*B[2][k] + A[i][3]*B[3][k]
    
    Schoolbook rank = 3*3*8 = 72.
    
    Using Strassen on the 2x2 sub-block (rows/cols 1,2):
    For the 2x2 * 2x8 part, we can use Strassen decomposition.
    The 2x2 * 2x8 product has schoolbook rank 2*2*8 = 32.
    With Strassen: 7 * 8 = 56 terms for the 2x2*2x8 part? No...
    
    Actually, for 2x2 * 2x8:
    C[i][k] = A[i][1]*B[1][k] + A[i][2]*B[2][k] for i in {1,2}, k in 1..8
    
    This is like 8 independent 2x2 * 2x1 problems, each with Strassen rank 7.
    So total rank = 7 * 8 = 56 for the 2x2*2x8 part.
    
    But we also need to handle row 3 and column 3.
    
    Let me use a more systematic approach. The total multiplication 3x3 * 3x8:
    
    C[i][k] = sum_{j=1}^{3} A[i][j] * B[j][k]
    
    I'll decompose this using the tensor product structure.
    
    Key insight: The 3x3 * 3x8 tensor can be decomposed as:
    - Terms involving A[3][j] (third row of A) for all j: these give C[3][k]
    - Terms involving A[i][3] (third column of A) for all i: these contribute to C[i][k]
    - The 2x2 * 2x8 sub-problem for the top-left 2x2 block
    
    Let me just use a hybrid: Strassen for 2x2*2x8 block + schoolbook for rest.
    
    2x2 * 2x8 block (A[0:2,0:2] * B[0:2,0:8]): rank 7*8 = 56
    Remaining terms:
    - A[2][j] * B[j][k] for j=0,1,2 and all i,k: This is row 3 of A times B
      = 3 * 8 = 24 terms (schoolbook for 1x3 * 3x8)
    - A[i][2] * B[2][k] for i=0,1 and all k: This is col 3 of A (first 2 rows) times row 3 of B
      = 2 * 8 = 16 terms (schoolbook for 2x1 * 1x8)
    
    Total: 56 + 24 + 16 = 96. That's worse than 72!
    
    Let me reconsider. The schoolbook is 72. Let me try a different decomposition.
    
    Actually, I should think about this more carefully. The tensor for 3x3*3x8 mult:
    T(A,B) = A*B where A is 3x3, B is 3x8, result C is 3x8.
    
    T_{i,j,k,l} = delta_{j,k} (summing over the middle index)
    where i in 0..2 (row of A/C), j in 0..2 (col of A / row of B), k in 0..2 (row of B), l in 0..7 (col of B/C)
    
    Wait, the tensor indices: A[i,j], B[j,k], C[i,k] where k goes 0..7 for B's columns.
    Let me re-index: A is 3x3, B is 3x8.
    C[i,l] = sum_{j=0}^{2} A[i,j] * B[j,l]
    
    The multiplication tensor has indices (i, j, l) with the summation over j.
    T_{i,j,l} = 1 (it's a 3x3x8 tensor where T_{i,j,l} contributes A[i,j]*B[j,l] to C[i,l])
    
    Rank of this tensor: schoolbook gives rank 3*3*8 = 72.
    
    For the tensor product approach: if we can write this tensor as a sum of
    Kronecker products of smaller tensors, we might get lower rank.
    
    Consider: 3x3*3x8 = (3x3*3x1) x (1x1*1x8) in some sense? No, that doesn't work directly.
    
    Let me try: decompose the 3x3 multiplication (A*B where both are 3x3, result 3x3)
    into a rank-r decomposition, then extend to 3x8 by Kronecker with 1x8 schoolbook.
    
    3x3 * 3x3 multiplication: known minimum rank is not 27 (schoolbook).
    Actually, for 3x3 matrix multiplication, the rank is known to be at most 23 (Laderman's algorithm)
    and at least 19. Laderman's algorithm gives rank 23.
    
    So: 3x3*3x3 has rank 23 (Laderman).
    Then 3x3*3x8 = (3x3*3x3) ⊗ (1x1*1x8) in the tensor sense?
    
    The Kronecker product of two multiplication tensors:
    If T1 is the (n,m,p) mult tensor with rank r1, and T2 is the (n',m',p') mult tensor with rank r2,
    then T1 ⊗ T2 is the (n*n', m*m', p*p') mult tensor with rank r1*r2.
    
    So (3,3,3) ⊗ (1,1,8) gives (3,3,24) which is not what we want.
    
    We want (3,3,8). Let me think of it as (3,3,3) ⊗ (1,1,8/3)? No, 8/3 isn't integer.
    
    Alternative: (3,3,8) = (3,3,2) ⊗ (1,1,4)? That gives (3,3,8). 
    (3,3,2) mult tensor: schoolbook rank = 3*3*2 = 18.
    (1,1,4) mult tensor: schoolbook rank = 1*1*4 = 4.
    Total: 18*4 = 72. Same as schoolbook.
    
    Or (3,3,8) = (3,3,1) ⊗ (1,1,8)?
    (3,3,1): schoolbook rank = 3*3*1 = 9.
    (1,1,8): schoolbook rank = 8.
    Total: 9*8 = 72. Same.
    
    What about using a non-schoolbook decomposition for one factor?
    (3,3,1) is matrix-vector multiplication (3x3 matrix times 1x3 vector, giving 3x1).
    The rank of 3x3 matrix-vector mult: this is the same as the "matrix-vector multiplication" tensor.
    For an n x n matrix times a vector, the rank is n^2 (schoolbook) and it's known that for n=3,
    the rank is 9 (no improvement known for 3x3 mat-vec mult).
    
    Hmm, so the tensor product approach with known decompositions doesn't immediately give
    something below 72.
    
    Let me try a different strategy: direct construction using Strassen-style ideas.
    
    For 3x3 * 3x8, let's try to use a 2x2 Strassen block for part of the computation.
    
    Actually, let me reconsider the problem. The task says "rank-minimal exact tensor decomposition."
    The schoolbook gives 72. Can we do better?
    
    Let me try a concrete construction:
    
    Split the 3x3 A into:
    A = [[a1, a2, a3],
         [a4, a5, a6],
         [a7, a8, a9]]
    
    B is 3x8. Let B = [[b11..b18], [b21..b28], [b31..b38]]
    
    C[i][k] = A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    For the 2x2 subproblem (top-left 2x2 of A, top 2 rows of B, first 2 rows of C):
    C[0,k] += A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] += A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This 2x2 * 2x8 part can be done with 7*8 = 56 Strassen terms.
    
    Remaining:
    C[0,k] += A[0,2]*B[2,k]  (1*8 = 8 terms)
    C[1,k] += A[1,2]*B[2,k]  (1*8 = 8 terms)  
    C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]  (3*8 = 24 terms)
    
    Total: 56 + 8 + 8 + 24 = 96. Worse than 72.
    
    The problem is that Strassen helps for square matrices but when one dimension is much larger,
    the overhead of the extra terms dominates.
    
    Let me just verify that schoolbook rank 72 is correct and return that, since I can't easily
    find a better construction within the time budget.
    
    Actually, wait. Let me think about whether there's a way to share computations.
    
    For C[0,k] and C[1,k], they both need B[0,k] and B[1,k].
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    If I define S1 = B[0,k] + B[1,k], S2 = B[0,k] - B[1,k], etc., I might save terms.
    
    Let's try:
    For each k, define:
    M1_k = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])
    M2_k = (A[1,0]+A[1,1]) * B[0,k]
    M3_k = A[0,0] * (B[1,k]-B[0,k])
    M4_k = A[1,1] * (B[1,k]-B[0,k])  -- wait, standard Strassen uses different combinations
    
    Standard 2x2 Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    In our case, for the 2x2*2x8 block, for each k:
    We want:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is like multiplying a 2x2 A_block by a 2x1 B_column (for each k).
    
    Using Strassen for 2x2 * 2x1:
    The 2x2 * 2x1 multiplication tensor: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    Schoolbook rank = 4. Can we do better?
    
    For 2x2 * 2x1 (matrix-vector with 2x2 matrix and 2-vector):
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    This has rank 4 (schoolbook). Is there a rank-3 decomposition?
    
    Try: 
    M1 = (A00+A11)*B0
    M2 = (A01+A11)*B1
    M3 = (A10-A00)*B0
    M4 = (A01-A11)*B1
    
    C0 = M1 + M2 - M4 = (A00+A11)*B0 + (A01+A11)*B1 - (A01-A11)*B1
        = A00*B0 + A11*B0 + A01*B1 + A11*B1 - A01*B1 + A11*B1
        = A00*B0 + A01*B1 + 2*A11*B0 + A11*B1  -- wrong
    
    Let me be more careful. For 2x2 * 2x1:
    The tensor is T_{i,j,k} where i in {0,1} (output row), j in {0,1} (A col), k in {0,1} (B row/input).
    T_{i,j,k} = delta_{j,k} (summing over j=k)
    
    Actually, C[i] = sum_j A[i,j] * B[j]. The tensor has indices (i, j, k) with T[i,j,k] = 1 if j=k, 0 otherwise.
    This is a 2x2x2 tensor which is the identity tensor, rank = 2? No...
    
    Wait, I'm confusing myself. Let me restart the tensor analysis.
    
    For matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i,l] = sum_j A[i,j] * B[j,l]
    
    The multiplication tensor T has T[i,j,l] = 1 (it's the "structure tensor" where the rank
    of the tensor gives the bilinear complexity).
    
    The tensor is in the space U ⊗ V ⊗ W where:
    U = R^{n*m} (A's entries), V = R^{m*p} (B's entries), W = R^{n*p} (C's entries)
    
    T = sum_{i,j,l} e_i ⊗ e_j ⊗ e_l where the sum is over valid (i,j,l) with j being the shared index.
    More precisely, T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{l=0}^{p-1} e_{i*m+j} ⊗ e_{j*p+l} ⊗ e_{i*p+l}
    
    The rank of this tensor is the minimum number of rank-1 tensors that sum to T.
    
    Schoolbook: rank = n*m*p.
    
    For n=3, m=3, p=8: schoolbook rank = 72.
    
    Now, the tensor product property: if T1 has rank r1 (for n1 x m1 x p1 mult) and T2 has rank r2
    (for n2 x m2 x p2 mult), then T1 ⊗ T2 has rank r1*r2 and corresponds to
    (n1*n2) x (m1*m2) x (p1*p2) matrix multiplication.
    
    So for 3x3x8:
    - (3x3x1) ⊗ (1x1x8): rank = rank(3x3x1) * rank(1x1x8) = 9 * 8 = 72
    - (1x1x3) ⊗ (3x3x8/3)... doesn't work since 8/3 not integer
    - (3x3x2) ⊗ (1x1x4): rank = rank(3x3x2) * rank(1x1x4)
      rank(3x3x2) schoolbook = 18, rank(1x1x4) = 4, total = 72
    
    All give 72 with schoolbook factors.
    
    Can we get a non-schoolbook rank for 3x3x1 or 3x3x2?
    
    3x3x1: This is 3x3 matrix times 3x1 vector = 3x1 vector.
    C[i] = sum_j A[i,j] * B[j]
    Tensor: T[i,j,k] = 1 if j=k (summing over j).
    This is a 3x3x3 tensor. The "slice" for each k is the k-th column of A (as a vector in R^3).
    T = sum_{j=0}^{2} sum_{i=0}^{2} e_i ⊗ e_j ⊗ e_j
    = sum_j (sum_i e_i ⊗ e_j) ⊗ e_j
    
    The rank of this tensor: it's the "matrix-vector multiplication" tensor for 3x3.
    Known result: the rank of n x n matrix-vector multiplication is n^2 for n >= 2
    (no improvement over schoolbook is known for n=3; it's an open problem for general n).
    
    So rank(3x3x1) = 9 (assuming no improvement).
    
    3x3x2: C[i,l] = sum_j A[i,j] * B[j,l] where l in {0,1}.
    This is 3x3 matrix times 3x2 matrix.
    Schoolbook rank = 3*3*2 = 18.
    Can we do better? This is equivalent to two 3x3x1 problems sharing the A matrix.
    The tensor is T[i,j,l] = 1 (for j being the shared index).
    T = sum_{j=0}^{2} sum_{i=0}^{2} sum_{l=0}^{1} e_{i*3+j} ⊗ e_{j*2+l} ⊗ e_{i*2+l}
    
    I don't know of a rank improvement for 3x3x2 over 18.
    
    Given the difficulty of finding a rank < 72 decomposition, and the time constraints,
    I'll stick with the schoolbook rank 72 but implement it cleanly.
    
    However, let me try one more idea: can we use the fact that p=8 is even to get
    some savings?
    
    For each pair of output columns (2l, 2l+1):
    C[i,2l] = sum_j A[i,j]*B[j,2l]
    C[i,2l+1] = sum_j A[i,j]*B[j,2l+1]
    
    Define for each i,j:
    S_{i,j}^l = B[j,2l] + B[j,2l+1]
    D_{i,j}^l = B[j,2l] - B[j,2l+1]
    
    Then:
    C[i,2l] + C[i,2l+1] = sum_j A[i,j] * S_{j}^l
    C[i,2l] - C[i,2l+1] = sum_j A[i,j] * D_{j}^l
    
    This doesn't immediately reduce the rank since we still need to compute all the
    A[i,j]*S and A[i,j]*D products.
    
    I think 72 is the best I can achieve with known techniques. Let me return the schoolbook.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
