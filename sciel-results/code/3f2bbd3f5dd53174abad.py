import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np


def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if s != 1:
                    return False
    return True


def _build_rank21_decomposition(n=3, m=3, p=3):
    """
    Explicit rank-21 decomposition for 3x3 x 3x3 matrix multiplication.
    
    We partition A and B as:
      A = [[A22, A21], [A12, A11]]  where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
      B = [[B22, B21], [B12, B11]]  where B22 is 2x2, B21 is 2x1, B12 is 1x2, B11 is 1x1
    
    The product C = A*B has blocks:
      C22 = A22*B22 + A21*B12  (2x2)
      C21 = A22*B21 + A21*B11  (2x1)
      C12 = A12*B22 + A11*B12  (1x2)
      C11 = A12*B21 + A11*B11  (1x1)
    
    We use Strassen's 7-term formula for the 2x2 blocks:
      M1 = (A11 + A22)(B11 + B22)
      M2 = (A21 + A22)B11
      M3 = A11(B21 - B22)
      M4 = A22(B12 - B11)
      M5 = (A11 + A12)B22
      M6 = (A21 - A11)(B11 + B12)
      M7 = (A12 - A22)(B21 + B22)
    
    With C11 = M1 + M4 - M5 + M7
         C12 = M3 + M5
         C21 = M2 + M4
         C22 = M1 - M2 + M3 + M6
    
    In our indexing, for the 2x2 block:
      A11 corresponds to element (0,0), A12 to (0,1), A21 to (1,0), A22 to (1,1)
      Similarly for B.
    
    Global indices (i, j) for A[i,j]:
      i in {0,1,2}, j in {0,1,2}
      A22: (0,0),(0,1),(1,0),(1,1) -> rows 0,1; cols 0,1
      A21: (0,2),(1,2) -> rows 0,1; col 2
      A12: (2,0),(2,1) -> row 2; cols 0,1
      A11: (2,2) -> row 2; col 2
    
    So mapping to Strassen variables for the 2x2 block (top-left 2x2):
      A11_strassen = A[0,0]
      A12_strassen = A[0,1]
      A21_strassen = A[1,0]
      A22_strassen = A[1,1]
    
    Similarly for B:
      B11_strassen = B[0,0]
      B12_strassen = B[0,1]
      B21_strassen = B[1,0]
      B22_strassen = B[1,1]
    
    Now we build the rank-21 decomposition.
    
    The 7 Strassen terms contribute to C22 (the 2x2 block at rows 0,1 and cols 0,1).
    Then we add terms for the cross-block products.
    
    Let's define the 7 Strassen terms explicitly with indices.
    For term c, we need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    The Strassen terms compute linear combinations of products. Each term M_t is a product
    of a linear form in A and a linear form in B. In tensor decomposition, each term
    corresponds to one column c where U, V, W have non-zero entries.
    
    For M1 = (A00 + A11)(B00 + B11):
      This contributes to C[i,k] for i,k in {0,1} (the 2x2 block).
      Specifically, M1 appears in C00, C10, C01, C11 with coefficients.
      From Strassen: C11 = M1 + M4 - M5 + M7, C22 = M1 - M2 + M3 + M6
      Wait, I need to be careful with the mapping.
      
    Actually, let me use a different approach. I'll construct the decomposition by
    explicitly specifying which (i,j,k) triples each column contributes to.
    
    For the Strassen 2x2 formula applied to the top-left 2x2 block:
    
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
      U: (0,0)=1, (1,1)=1
      V: (0,0)=1, (1,1)=1
      W: contributes to C[0,0], C[0,1], C[1,0], C[1,1]
      From Strassen: C[0,0] gets +M1, C[0,1] gets +M1? No.
      
    Let me look up the exact Strassen formulas again:
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
      
    Where C11 is top-left, C12 is top-right, C21 is bottom-left, C22 is bottom-right.
    
    So for the 2x2 block (our A22, B22, C22 in the 3x3 partition):
      C22[0,0] (i.e., C[0,0]) = M1 + M4 - M5 + M7
      C22[0,1] (i.e., C[0,1]) = M3 + M5
      C22[1,0] (i.e., C[1,0]) = M2 + M4
      C22[1,1] (i.e., C[1,1]) = M1 - M2 + M3 + M6
    
    Now I need to translate each M_t into U, V, W vectors.
    
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
      U column: U[0*3+0]=U[0]=1, U[1*3+1]=U[4]=1
      V column: V[0*3+0]=V[0]=1, V[1*3+1]=V[4]=1
      W column: W[0*3+0]=W[0]=1 (for C[0,0]), W[0*3+1]=W[1]=1 (for C[0,1])? 
      
    Wait, W[k*n+i] gives the coefficient for C[i,k].
    For M1, it contributes to C[0,0], C[0,1], C[1,0], C[1,1] based on the Strassen formulas.
    
    Actually, each M_t is a single scalar product. In the tensor decomposition,
    each column c corresponds to one bilinear term. The term M1 is:
      (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    This is a rank-1 term in the (A, B) space, but it contributes to multiple (C) entries.
    
    In the tensor decomposition T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c],
    for a fixed column c, the contribution to C[i,k] (via element j of the intermediate) is:
      sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    For M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    This means:
      U[i,j,c] is non-zero when (i,j) is (0,0) or (1,1)
      V[j,k,c] is non-zero when (j,k) is (0,0) or (1,1)
      W[k,i,c] provides the coefficient for C[i,k]
    
    Let's define:
      U[0, c] = 1   (i=0, j=0)
      U[4, c] = 1   (i=1, j=1)
      V[0, c] = 1   (j=0, k=0)
      V[4, c] = 1   (j=1, k=1)
    
    Then for C[i,k]:
      sum_j U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For i=0, k=0:
      j=0: U[0,c]*V[0,c]*W[0,c] = 1*1*W[0,c]
      j=1: U[1,c]*V[1,c]*W[0,c] = 0*...*W[0,c] = 0
      So contribution = W[0,c]
    
    For i=0, k=1:
      j=0: U[0,c]*V[1,c]*W[3,c] = 1*0*W[3,c] = 0  (V[1,c] is V[j=1,k=1] but we need V[0,c] for j=0,k=1)
      Wait, V[j*p+k] = V[j*3+k].
      V[0,c] is j=0,k=0. V[1,c] is j=0,k=1. V[3,c] is j=1,k=0. V[4,c] is j=1,k=1.
      
      So for i=0, k=1:
      j=0: U[0,c]*V[1,c]*W[3,c] = 1*0*W[3,c] = 0
      j=1: U[1,c]*V[4,c]*W[3,c] = 0*1*W[3,c] = 0
      Contribution = 0. But M1 should contribute to C[0,1] with coefficient 1!
    
    I see the issue. The Strassen term M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) is not directly
    a single tensor term. It's a product of two linear forms, which in the bilinear complexity
    sense corresponds to multiple tensor terms or requires a different interpretation.
    
    Let me reconsider. In bilinear complexity, each term is of the form:
      (sum_i a_i * A_i) * (sum_j b_j * B_j)
    which contributes to the product C = A*B with coefficients determined by how the
    A_i and B_j map to C entries.
    
    For M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    This is one bilinear term. In the tensor decomposition, it corresponds to a rank-1
    tensor in the (A, B, C) space. The U, V, W vectors encode this.
    
    U[i,j,c] = coefficient of A[i,j] in the first linear form = 1 if (i,j)=(0,0) or (1,1), else 0
    V[j,k,c] = coefficient of B[j,k] in the second linear form = 1 if (j,k)=(0,0) or (1,1), else 0
    
    Then the contribution to C[i,k] from this term is:
      sum_j U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For this to equal the correct coefficient, we need to choose W[k,i,c] such that:
      sum_j U[i,j,c] * V[j,k,c] * W[k,i,c] = coefficient of C[i,k] from M1
    
    For M1, the coefficients are:
      C[0,0]: +1 (from C11 = M1 + ...)
      C[0,1]: +1? No, C12 = M3 + M5, so M1 doesn't appear in C[0,1].
      
    Wait, I'm confusing myself. Let me be very explicit.
    
    The 2x2 Strassen formulas:
      C11 = M1 + M4 - M5 + M7   (top-left of C)
      C12 = M3 + M5              (top-right of C)
      C21 = M2 + M4              (bottom-left of C)
      C22 = M1 - M2 + M3 + M6   (bottom-right of C)
    
    Where C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1].
    
    So:
      M1 contributes: +1 to C[0,0], +1 to C[1,1]
      M2 contributes: +1 to C[1,0], -1 to C[1,1]
      M3 contributes: +1 to C[0,1], +1 to C[1,1]
      M4 contributes: +1 to C[0,0], +1 to C[1,0]
      M5 contributes: -1 to C[0,0], +1 to C[0,1]
      M6 contributes: +1 to C[1,1]
      M7 contributes: +1 to C[0,0]
    
    Now for each M_t, I define U and V, and then W is determined by the required contributions.
    
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
      U[0,c]=1, U[4,c]=1
      V[0,c]=1, V[4,c]=1
      Required: C[0,0] += 1, C[1,1] += 1
      Contribution to C[i,k] = sum_j U[i,j,c]*V[j,k,c]*W[k,i,c]
      
      For C[0,0] (i=0,k=0):
        j=0: U[0,c]*V[0,c]*W[0,c] = 1*1*W[0,c]
        j=1: U[1,c]*V[3,c]*W[0,c] = 0*...*W[0,c] = 0
        => W[0,c] = 1
      
      For C[1,1] (i=1,k=1):
        j=0: U[0,c]*V[1,c]*W[3,c] = 0*...*W[3,c] = 0  (U[0,c] is for i=0, not i=1)
        Wait, U[i*m+j] = U[i*3+j].
        For i=1, j=0: U[3,c] = 0
        For i=1, j=1: U[4,c] = 1
        j=0: U[3,c]*V[1,c]*W[3,c] = 0*...*W[3,c] = 0
        j=1: U[4,c]*V[4,c]*W[3,c] = 1*1*W[3,c]
        => W[3,c] = 1  (W[k*n+i] = W[1*3+1] = W[4]? No, W[k*n+i] = W[k*3+i])
        
        For C[1,1], i=1, k=1: W[k*n+i] = W[1*3+1] = W[4]
        So W[4,c] = 1
      
      For C[0,1] (i=0,k=1):
        j=0: U[0,c]*V[1,c]*W[3,c] = 1*0*W[3,c] = 0  (V[1,c] is j=0,k=1)
        j=1: U[1,c]*V[4,c]*W[3,c] = 0*1*W[3,c] = 0
        => 0, which is correct (M1 doesn't contribute to C[0,1])
      
      For C[1,0] (i=1,k=0):
        j=0: U[3,c]*V[0,c]*W[0,c] = 0*1*W[0,c] = 0
        j=1: U[4,c]*V[3,c]*W[0,c] = 1*0*W[0,c] = 0
        => 0, correct
      
      So for M1: U[0,c]=1, U[4,c]=1, V[0,c]=1, V[4,c]=1, W[0,c]=1, W[4,c]=1
    
    Let me verify this pattern for M2.
    
    M2 = (A[1,0]+A[1,1])*(B[0,0])
      U[3,c]=1, U[4,c]=1  (i=1, j=0 and i=1, j=1)
      V[0,c]=1  (j=0, k=0)
      Required: C[1,0] += 1, C[1,1] += -1
      
      For C[1,0] (i=1,k=0):
        j=0: U[3,c]*V[0,c]*W[0,c] = 1*1*W[0,c]
        j=1: U[4,c]*V[3,c]*W[0,c] = 1*0*W[0,c] = 0
        => W[0,c] = 1
      
      For C[1,1] (i=1,k=1):
        j=0: U[3,c]*V[1,c]*W[4,c] = 1*0*W[4,c] = 0
        j=1: U[4,c]*V[4,c]*W[4,c] = 1*0*W[4,c] = 0  (V[4,c] is j=1,k=1, but V[0,c] is j=0,k=0)
        Wait, V[j,k,c] for M2 is only V[0,c]=1 (j=0,k=0). So V[4,c]=0.
        => 0, but we need -1!
      
      Problem: M2 = (A[1,0]+A[1,1])*B[0,0] only has B[0,0] in the second form.
      So V[0,c]=1, V[1,c]=0, V[3,c]=0, V[4,c]=0.
      
      For C[1,1], we need contribution -1. But with only V[0,c]=1:
        j=0: U[3,c]*V[0,c]*W[4,c] = 1*1*W[4,c]
        j=1: U[4,c]*V[4,c]*W[4,c] = 1*0*W[4,c] = 0
        => W[4,c] = -1
      
      So for M2: U[3,c]=1, U[4,c]=1, V[0,c]=1, W[0,c]=1, W[4,c]=-1
    
    This is getting complex. Let me just hardcode the rank-21 decomposition based on known results.
    
    Actually, let me use a simpler approach: I'll use the known rank-23 decomposition or
    try to find a rank-21 one via search, but with a much better initialization.
    
    Given the complexity, let me just return the schoolbook rank-27 for now, since the
    random search in the original code also returned 27. The improvement would come from
    a better search strategy, but constructing an explicit low-rank decomposition is
    non-trivial.
    
    Let me try a different approach: use the explicit rank-21 construction from literature.
    """
    n, m, p = 3, 3, 3
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 for the top-left 2x2 block (indices 0,1 in i and k)
    # M1 = (A00+A11)(B00+B11)
    c = 0
    U[0][c] = 1; U[4][c] = 1
    V[0][c] = 1; V[4][c] = 1
    W[0][c] = 1; W[4][c] = 1
    
    # M2 = (A10+A11)(B00)
    c = 1
    U[3][c] = 1; U[4][c] = 1
    V[0][c] = 1
    W[0][c] = 1; W[4][c] = -1
    
    # M3 = (A00)(B10-B11)
    c = 2
    U[0][c] = 1
    V[3][c] = 1; V[4][c] = -1
    W[1][c] = 1; W[4][c] = 1
    
    # M4 = (A11)(B01-B00)
    c = 3
    U[4][c] = 1
    V[1][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[3][c] = 1
    
    # M5 = (A00+A01)(B11)
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[4][c] = 1
    W[0][c] = -1; W[1][c] = 1
    
    # M6 = (A10-A00)(B00+B01)
    c = 5
    U[3][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[4][c] = 1
    
    # M7 = (A01-A11)(B10+B11)
    c = 6
    U[1][c] = 1; U[4][c] = -1
    V[3][c] = 1; V[4][c] = 1
    W[0][c] = 1
    
    # Now the cross-block terms:
    # C22 (rows 0,1; cols 0,1) also gets A21*B12 where A21 is 2x1, B12 is 1x2
    # A21: A[0,2], A[1,2]; B12: B[2,0], B[2,1]
    # This is a rank-2 term (2x1 times 1x2 = 2x2, but actually it's an outer product scaled)
    # C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1], C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
    
    # Term for A[0,2]*B[2,0] -> C[0,0]
    c = 7
    U[2][c] = 1  # i=0, j=2
    V[6][c] = 1  # j=2, k=0
    W[0][c] = 1  # k=0, i=0
    
    # Term for A[0,2]*B[2,1] -> C[0,1]
    c = 8
    U[2][c] = 1
    V[7][c] = 1  # j=2, k=1
    W[1][c] = 1  # k=1, i=0
    
    # Term for A[1,2]*B[2,0] -> C[1,0]
    c = 9
    U[5][c] = 1  # i=1, j=2
    V[6][c] = 1
    W[3][c] = 1  # k=0, i=1
    
    # Term for A[1,2]*B[2,1] -> C[1,1]
    c = 10
    U[5][c] = 1
    V[7][c] = 1
    W[4][c] = 1
    
    # C21 (rows 0,1; col 2): A22*B21 + A21*B11
    # A22*B21: 2x2 times 2x1 -> rank 4
    # A21*B11: 2x1 times 1x1 -> rank 2
    
    # A[0,0]*B[0,2] -> C[0,2]
    c = 11
    U[0][c] = 1
    V[2][c] = 1  # j=0, k=2
    W[6][c] = 1  # k=2, i=0
    
    # A[0,1]*B[1,2] -> C[0,2]
    c = 12
    U[1][c] = 1
    V[5][c] = 1  # j=1, k=2
    W[6][c] = 1
    
    # A[1,0]*B[0,2] -> C[1,2]
    c = 13
    U[3][c] = 1
    V[2][c] = 1
    W[8][c] = 1  # k=2, i=1
    
    # A[1,1]*B[1,2] -> C[1,2]
    c = 14
    U[4][c] = 1
    V[5][c] = 1
    W[8][c] = 1
    
    # A[0,2]*B[2,2] -> C[0,2]
    c = 15
    U[2][c] = 1
    V[8][c] = 1  # j=2, k=2
    W[6][c] = 1
    
    # A[1,2]*B[2,2] -> C[1,2]
    c = 16
    U[5][c] = 1
    V[8][c] = 1
    W[8][c] = 1
    
    # C12 (row 2; cols 0,1): A12*B22 + A11*B12
    # A12*B22: 1x2 times 2x2 -> rank 4
    # A11*B12: 1x1 times 1x2 -> rank 2
    
    # A[2,0]*B[0,0] -> C[2,0]
    c = 17
    U[6][c] = 1  # i=2, j=0
    V[0][c] = 1
    W[0][c] = 1  # k=0, i=2
    
    # A[2,0]*B[0,1] -> C[2,1]
    c = 18
    U[6][c] = 1
    V[1][c] = 1  # j=0, k=1
    W[1][c] = 1
    
    # A[2,1]*B[1,0] -> C[2,0]
    c = 19
    U[7][c] = 1  # i=2, j=1
    V[3][c] = 1
    W[0][c] = 1
    
    # A[2,1]*B[1,1] -> C[2,1]
    c = 20
    U[7][c] = 1
    V[4][c] = 1
    W[1][c] = 1
    
    # A[2,2]*B[2,0] -> C[2,0]
    # A[2,2]*B[2,1] -> C[2,1]
    # But wait, we only have 21 columns. Let me recount.
    
    # I have 7 Strassen + 4 (A21*B12) + 6 (C21 terms) + 4 (C12 partial) = 21 so far,
    # but I'm missing A11*B12 and A12*B21, A11*B11.
    
    # This is getting too complex to do by hand. Let me just verify if this partial
    # construction is correct, and if not, fall back to rank 27.
    
    return None  # This explicit construction is incomplete/incorrect, so return None


def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed schoolbook rank-27 decomposition
    rank_schoolbook = n * m * p  # 27
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try the explicit rank-21 construction
    r21 = _build_rank21_decomposition(n, m, p)
    if r21 is not None:
        U, V, W = r21
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = 21
    
    # Try to find lower-rank decompositions via search
    # Allocate remaining time for search
    elapsed = time.time() - start_time
    remaining = timeout - 3 - elapsed
    if remaining > 10:
        # Try rank 23, 22, 21, 20
        for target_rank in range(23, min(best_rank, 18), -1):
            if time.time() - start_time > timeout - 3:
                break
            
            budget = remaining / 5.0
            result = _try_rank(target_rank, n, m, p, time_budget=budget, seed=0)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best_result


def _try_rank(r, n=3, m=3, p=3, time_budget=10.0, seed=0):
    """
    Attempt to find an exact rank-r decomposition using structured initialization
    and targeted local search with NumPy-accelerated error computation.
    Returns (U, V, W) or None.
    """
    start = time.time()
    import random
    rng = random.Random(seed)
    
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    
    def compute_errors():
        """Compute error tensor E[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1"""
        errors = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    errors[i, j, k] = s - 1
        return errors
    
    def total_error(errors):
        return np.sum(errors * errors)
    
    # Initialize with a pattern
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    rng.shuffle(triples)
    for c, (i, j, k) in enumerate(triples[:r]):
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
    
    errors = compute_errors()
    current_err = total_error(errors)
    
    # Local search phase
    no_improve_count = 0
    max_steps = 10000
    
    while no_improve_count < 300 and max_steps > 0:
        if time.time() - start > time_budget:
            break
        max_steps -= 1
        
        factor = rng.choice(['U', 'V', 'W'])
        if factor == 'U':
            row = rng.randrange(n * m)
        elif factor == 'V':
            row = rng.randrange(m * p)
        else:
            row = rng.randrange(p * n)
        col = rng.randrange(r)
        
        M = {'U': U, 'V': V, 'W': W}[factor]
        old_val = M[row, col]
        new_val = rng.choice([0, 1, -1])
        if new_val == old_val:
            continue
        
        M[row, col] = new_val
        
        errors = compute_errors()
        new_err = total_error(errors)
        
        if new_err < current_err:
            current_err = new_err
            no_improve_count = 0
        else:
            M[row, col] = old_val
            no_improve_count += 1
        
        if current_err == 0:
            break
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    return None
# EVOLVE-BLOCK-END
