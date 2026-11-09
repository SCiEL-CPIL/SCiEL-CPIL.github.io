"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

# Deadline guard
_DEADLINE = 150.0  # seconds
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Uses block partitioning: 4 = 2+2, 5 = 2+3.
    The 2x2x2 blocks use Strassen (rank 7 instead of 8).
    The 3x3x3 blocks and cross-terms use schoolbook.
    
    Block structure of the (4x5) * (5x5) product:
    - A is 4x5, partitioned as A = [A_22, A_23; A_22', A_23'] where rows split 2+2, cols split 2+3
    - B is 5x5, partitioned as B = [B_22, B_23; B_32, B_33] where rows split 2+3, cols split 2+3
    
    Wait, let me reconsider. The matrix multiplication tensor for (n,m,p) = (4,5,5):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A is 4x5 (n x m), B is 5x5 (m x p), C is 4x5 (n x p).
    
    Partition:
    - n = 4 = 2 + 2 (rows of A and C)
    - m = 5 = 2 + 3 (cols of A, rows of B)
    - p = 5 = 2 + 3 (cols of B and C)
    
    So:
    A = [A_11 (2x2), A_12 (2x3); A_21 (2x2), A_22 (2x3)]
    B = [B_11 (2x2), B_12 (2x3); B_21 (3x2), B_22 (3x3)]
    C = [C_11 (2x2), C_12 (2x3); C_21 (2x2), C_22 (2x3)]
    
    C_11 = A_11 * B_11 + A_12 * B_21  (2x2 * 2x2 + 2x3 * 3x2)
    C_12 = A_11 * B_12 + A_12 * B_22  (2x2 * 2x3 + 2x3 * 3x3)
    C_21 = A_21 * B_11 + A_22 * B_21  (2x2 * 2x2 + 2x3 * 3x2)
    C_22 = A_21 * B_12 + A_22 * B_22  (2x2 * 2x3 + 2x3 * 3x3)
    
    Now let's count the schoolbook rank for each sub-product:
    - 2x2 * 2x2 -> 4 terms, but Strassen gives 7
    - 2x3 * 3x2 -> 6 terms
    - 2x2 * 2x3 -> 6 terms
    - 2x3 * 3x3 -> 9 terms
    
    For C_11: A_11*B_11 (2x2x2) + A_12*B_21 (2x3x2)
      Schoolbook: 4 + 6 = 10
      Strassen for A_11*B_11: 7, plus 6 for A_12*B_21 = 13
      But we can potentially share Strassen terms if the 2x2 submatrices are the same...
      Actually A_11 and A_21 are different rows of A, so they're different.
    
    Let me just count total schoolbook: 4*5*5 = 100.
    
    With Strassen on the four 2x2x2 sub-problems (A_11*B_11, A_21*B_11, A_11*B_12, A_21*B_12):
    Each 2x2x2 schoolbook is 4, Strassen is 7. So each saves... wait, 7 > 4. Strassen increases rank!
    
    Oh wait, I was confused. For 2x2x2, schoolbook is 4 and Strassen is 7. Strassen is for 2x2x2 matrix multiplication which has 8 scalar multiplications in schoolbook. Let me reclarify.
    
    For multiplying two 2x2 matrices: C = A * B where A, B, C are 2x2.
    Schoolbook: C[i][k] = sum_j A[i][j] B[j][k]. That's 2*2*2 = 8 scalar multiplications.
    Strassen: 7 scalar multiplications.
    
    So in the tensor decomposition, a 2x2x2 matrix multiplication tensor has rank 8 (schoolbook) or 7 (Strassen).
    
    Now back to our problem. The full tensor is for (4,5,5):
    C[i,k] = sum_j A[i,j] B[j,k], i in 0..3, j in 0..4, k in 0..4.
    Total schoolbook rank = 4*5*5 = 100.
    
    With block partitioning 4=2+2, 5=2+3:
    
    The sub-tensors are:
    1. A_11 (2x2) * B_11 (2x2) -> contributes to C_11 (2x2): this is a 2x2x2 MM tensor, rank 8 schoolbook, 7 Strassen
    2. A_12 (2x3) * B_21 (3x2) -> contributes to C_11 (2x2): rank 2*3*2 = 12 schoolbook
    3. A_21 (2x2) * B_11 (2x2) -> contributes to C_21 (2x2): rank 8 schoolbook, 7 Strassen
    4. A_22 (2x3) * B_21 (3x2) -> contributes to C_21 (2x2): rank 12 schoolbook
    5. A_11 (2x2) * B_12 (2x3) -> contributes to C_12 (2x3): rank 2*2*3 = 12 schoolbook
    6. A_12 (2x3) * B_22 (3x3) -> contributes to C_12 (2x3): rank 2*3*3 = 18 schoolbook
    7. A_21 (2x2) * B_12 (2x3) -> contributes to C_22 (2x3): rank 12 schoolbook
    8. A_22 (2x3) * B_22 (3x3) -> contributes to C_22 (2x3): rank 18 schoolbook
    
    Total schoolbook: 8+12+8+12+12+18+12+18 = 100. Checks out.
    
    With Strassen on the 2x2x2 parts:
    1. 7 (Strassen) + 12 + 7 (Strassen) + 12 + 12 + 18 + 12 + 18 = 98
    
    So we save 2 by using Strassen on the four 2x2x2 sub-problems.
    
    But wait - can we do better? The 3x3x3 sub-problems (B_22 appears in two places, and B_21 appears in two places) might allow sharing.
    
    Actually, let me think about this differently. The key insight is that B_11 (2x2) is used in both A_11*B_11 and A_21*B_11. Similarly B_12 (2x3) is used in A_11*B_12 and A_21*B_12. And B_21 (3x2) is used in A_12*B_21 and A_22*B_21. And B_22 (3x3) is used in A_12*B_22 and A_22*B_22.
    
    This suggests we could potentially use a shared decomposition for the "right factor" blocks. But the left factors are different (A_11 vs A_21, A_12 vs A_22), so direct sharing isn't straightforward.
    
    However, we can use the fact that for computing [A_11; A_21] * B_11 (a 4x2 * 2x2 = 4x2 product), we can use a single decomposition. The tensor for (4,2,2) matrix multiplication has schoolbook rank 4*2*2 = 16. Can we do better?
    
    Similarly [A_12; A_22] * B_21 is (4,3,2) with schoolbook rank 4*3*2 = 24.
    [A_11; A_21] * B_12 is (4,2,3) with schoolbook rank 4*2*3 = 24.
    [A_12; A_22] * B_22 is (4,3,3) with schoolbook rank 4*3*3 = 36.
    
    Total: 16 + 24 + 24 + 36 = 100. Same as before.
    
    Now, for (4,2,2): we can partition the 4 rows as 2+2. The 2x2x2 parts can use Strassen.
    (4,2,2) = (2,2,2) + (2,2,2) in terms of row blocks.
    Schoolbook: 8 + 8 = 16. Strassen: 7 + 7 = 14. Save 2.
    
    For (4,3,2): partition rows as 2+2.
    (2,3,2) + (2,3,2). Schoolbook: 12 + 12 = 24. No obvious Strassen-like improvement for 2x3x2.
    
    For (4,2,3): partition rows as 2+2.
    (2,2,3) + (2,2,3). Schoolbook: 12 + 12 = 24. No obvious improvement.
    
    For (4,3,3): partition rows as 2+2.
    (2,3,3) + (2,3,3). Schoolbook: 18 + 18 = 36. No obvious improvement.
    
    So with Strassen on the (4,2,2) part, we get 14 + 24 + 24 + 36 = 98. Same as before.
    
    Can we do even better? Let me think about whether the 3x3 blocks can be decomposed more efficiently.
    
    For 3x3x3 matrix multiplication, the best known rank is 23 (Landsberg's bound is 19, but best known explicit is 23 via Strassen's 3x3x3 algorithm). Schoolbook is 27.
    
    Wait, actually for 3x3x3, Strassen's algorithm gives rank 23. Let me verify: Strassen's 3x3x3 decomposition uses 23 multiplications. Yes, that's correct.
    
    So if we can isolate a 3x3x3 sub-problem, we can save 4 (27-23=4).
    
    Looking at our decomposition:
    - B_22 (3x3) is used in A_12*B_22 (2x3x3, rank 18) and A_22*B_22 (2x3x3, rank 18).
    - Together, [A_12; A_22] * B_22 is (4,3,3) with rank 36.
    
    Can we decompose (4,3,3) more efficiently? Partition the 4 rows as 2+2:
    (2,3,3) + (2,3,3) = 18 + 18 = 36.
    
    Alternatively, think of it as computing two 2x3 matrices times a 3x3 matrix. There's no standard algorithm that beats schoolbook for this specific shape that I know of.
    
    What about B_21 (3x2)? It's used in A_12*B_21 (2x3x2, rank 12) and A_22*B_21 (2x3x2, rank 12).
    Together: (4,3,2) with rank 24.
    
    And B_11 (2x2) is used in A_11*B_11 (2x2x2, rank 8->7) and A_21*B_11 (2x2x2, rank 8->7).
    Together: (4,2,2) with rank 16->14.
    
    And B_12 (2x3) is used in A_11*B_12 (2x2x3, rank 12) and A_21*B_12 (2x2x3, rank 12).
    Together: (4,2,3) with rank 24.
    
    So the best I can do with this partitioning is 98.
    
    Let me try a different partitioning. What if I use 5 = 2+2+1 or 5 = 3+2?
    
    Actually, let me reconsider. The known best rank for 4x5x5 matrix multiplication... Let me think about what's achievable.
    
    Actually, for the purposes of this task, let me just implement the Strassen-based decomposition that gives rank 98, which is an improvement over 100.
    
    Let me construct the Strassen decomposition for 2x2x2:
    For C = A * B where A, B, C are 2x2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    In tensor decomposition terms, for a 2x2x2 MM tensor:
    U has shape (2*2, 7) = (4, 7)
    V has shape (2*2, 7) = (4, 7)
    W has shape (2*2, 7) = (4, 7)
    
    The indices are:
    U[i*m + j, c] where i,j in 0..1, c in 0..6
    V[j*p + k, c] where j,k in 0..1, c in 0..6
    W[k*n + i, c] where k,i in 0..1, c in 0..6
    
    Wait, the convention is W[k*n+i, r]. For 2x2x2, n=m=p=2.
    
    Let me write out the Strassen decomposition:
    
    For M1: (A00+A11)(B00+B11)
    U: A00 -> U[0*2+0][0] += 1, A11 -> U[1*2+1][0] += 1
    V: B00 -> V[0*2+0][0] += 1, B11 -> V[1*2+1][0] += 1
    W: C00 -> W[0*2+0][0] += 1, C01 -> W[1*2+0][0] += 1, C10 -> W[0*2+1][0] += 1, C11 -> W[1*2+1][0] += 1
    
    Wait, I need to be more careful. The tensor T[i,j,k] = delta (meaning C[i,k] = sum_j A[i,j]B[j,k]).
    
    In the decomposition T = sum_r U[:,r] V[:,r] W[:,r] (outer product), we have:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For Strassen M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11:
    This contributes to:
    C00 gets A00*B00 + A00*B11 (wait no)
    
    Let me redo this. C = A*B:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Strassen:
    M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    M2 = (A10+A11)*B00 = A10*B00 + A11*B00
    M3 = A00*(B01-B11) = A00*B01 - A00*B11
    M4 = A11*(B10-B00) = A11*B10 - A11*B00
    M5 = (A00+A01)*B11 = A00*B11 + A01*B11
    M6 = (A10-A00)*(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    M7 = (A01-A11)*(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Let me verify C00:
    M1 + M4 - M5 + M7
    = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
      + (A11*B10 - A11*B00)
      - (A00*B11 + A01*B11)
      + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00
      - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    = A00*B00 + A01*B10
    ✓
    
    Now for the tensor decomposition, each M_r gives:
    
    M1: U has 1 at (0,0) and (1,1) [A00, A11]; V has 1 at (0,0) and (1,1) [B00, B11]
         W: C00 gets +1, C01 gets... wait.
         
    Actually, in the decomposition, W[k*n+i, r] is the coefficient for C[i,k] in term r.
    
    For M1, it contributes to all four C entries:
    C00: +1, C01: 0, C10: 0, C11: +1
    Wait no. M1 is a single scalar. It's added to C00 with coefficient +1 and to C11 with coefficient +1.
    
    So W for M1: W[0*2+0] = 1 (C00), W[1*2+1] = 1 (C11). Others 0.
    
    For M2: contributes to C10 with +1.
    W[0*2+1] = 1 (C10). Others 0.
    
    For M3: contributes to C01 with +1.
    W[1*2+0] = 1 (C01). Others 0.
    
    For M4: contributes to C00 with +1 and C10 with +1.
    W[0*2+0] = 1, W[0*2+1] = 1.
    
    For M5: contributes to C00 with -1 and C01 with +1.
    W[0*2+0] = -1, W[1*2+0] = 1.
    
    For M6: contributes to C11 with +1.
    W[1*2+1] = 1.
    
    For M7: contributes to C00 with +1.
    W[0*2+0] = 1.
    
    Let me now build the full decomposition for 4x5x5 with rank 98.
    
    The four 2x2x2 sub-problems that use Strassen:
    1. A_11 (rows 0-1, cols 0-1 of A) * B_11 (rows 0-1, cols 0-1 of B) -> C_11 (rows 0-1, cols 0-1 of C)
    2. A_21 (rows 2-3, cols 0-1 of A) * B_11 (rows 0-1, cols 0-1 of B) -> C_21 (rows 2-3, cols 0-1 of C)
    3. A_11 (rows 0-1, cols 0-1 of A) * B_12 (rows 0-1, cols 2-4 of B) -> C_12 (rows 0-1, cols 2-4 of C)
       Wait, this is 2x2 * 2x3, not 2x2x2. Let me reconsider.
    
    Hmm, I made an error. Let me re-examine.
    
    A is 4x5, B is 5x5, C is 4x5.
    Partition: n=4=2+2, m=5=2+3, p=5=2+3.
    
    A blocks: A[0:2, 0:2] (2x2), A[0:2, 2:5] (2x3), A[2:4, 0:2] (2x2), A[2:4, 2:5] (2x3)
    B blocks: B[0:2, 0:2] (2x2), B[0:2, 2:5] (2x3), B[2:5, 0:2] (3x2), B[2:5, 2:5] (3x3)
    C blocks: C[0:2, 0:2] (2x2), C[0:2, 2:5] (2x3), C[2:4, 0:2] (2x2), C[2:4, 2:5] (2x3)
    
    C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
                 = (2x2)(2x2) + (2x3)(3x2)
    
    C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5]
                 = (2x2)(2x3) + (2x3)(3x3)
    
    C[2:4, 0:2] = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:5]*B[2:5, 0:2]
                 = (2x2)(2x2) + (2x3)(3x2)
    
    C[2:4, 2:5] = A[2:4, 0:2]*B[0:2, 2:5] + A[2:4, 2:5]*B[2:5, 2:5]
                 = (2x2)(2x3) + (2x3)(3x3)
    
    The 2x2x2 sub-problems are:
    - A[0:2,0:2] * B[0:2,0:2] -> part of C[0:2,0:2]
    - A[2:4,0:2] * B[0:2,0:2] -> part of C[2:4,0:2]
    
    These are the only two 2x2x2 sub-problems! The others are 2x2x3, 2x3x2, 2x3x3.
    
    So with Strassen, we replace two 2x2x2 (rank 8 each) with Strassen (rank 7 each), saving 2 total.
    Rank: 100 - 2 = 98.
    
    Let me now construct this.
    
    For the first Strassen (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]):
    Local indices: i_local in 0..1, j_local in 0..1, k_local in 0..1
    Global: i = i_local, j = j_local, k = k_local
    
    For the second Strassen (A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]):
    Local indices: i_local in 0..1, j_local in 0..1, k_local in 0..1
    Global: i = 2 + i_local, j = j_local, k = k_local
    
    The remaining schoolbook terms:
    - A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2]: i in 0..1, j in 2..4, k in 0..1
    - A[2:4,2:5] * B[2:5,0:2] -> C[2:4,0:2]: i in 2..3, j in 2..4, k in 0..1
    - A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]: i in 0..1, j in 0..1, k in 2..4
    - A[2:4,0:2] * B[0:2,2:5] -> C[2:4,2:5]: i in 2..3, j in 0..1, k in 2..4
    - A[0:2,2:5] * B[2:5,2:5] -> C[0:2,2:5]: i in 0..1, j in 2..4, k in 2..4
    - A[2:4,2:5] * B[2:5,2:5] -> C[2:4,2:5]: i in 2..3, j in 2..4, k in 2..4
    
    Let me count:
    Strassen 1: 7
    Strassen 2: 7
    Schoolbook: 2*3*2 + 2*3*2 + 2*2*3 + 2*2*3 + 2*3*3 + 2*3*3 = 12+12+12+12+18+18 = 84
    Total: 7+7+84 = 98. ✓
    
    Now let me code this up.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 98
    V = [[0] * rank for _ in range(m * p)]  # 25 x 98
    W = [[0] * rank for _ in range(p * n)]  # 20 x 98
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    def add_strassen(i_off, j_off, k_off, start_col):
        """Add Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2] -> C[i_off:i_off+2, k_off:k_off+2]"""
        # M1 = (A00+A11)(B00+B11)
        U[i_off * m + j_off][start_col] = 1
        U[(i_off+1) * m + (j_off+1)][start_col] = 1
        V[j_off * p + k_off][start_col] = 1
        V[(j_off+1) * p + (k_off+1)][start_col] = 1
        W[k_off * n + i_off][start_col] = 1       # C00
        W[(k_off+1) * n + (i_off+1)][start_col] = 1  # C11
        col = start_col + 1
        
        # M2 = (A10+A11)*B00
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1  # C10
        col += 1
        
        # M3 = A00*(B01-B11)
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1  # C01
        col += 1
        
        # M4 = A11*(B10-B00)
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1       # C00
        W[k_off * n + (i_off+1)][col] = 1   # C10
        col += 1
        
        # M5 = (A00+A01)*B11
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1      # C00
        W[(k_off+1) * n + i_off][col] = 1   # C01
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1  # C11
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1  # C00
        col += 1
        
        return col
    
    # Strassen 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen(0, 0, 0, col)
    
    # Strassen 2: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    col = add_strassen(2, 0, 0, col)
    
    # Schoolbook terms
    # A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A[2:4, 2:5] * B[2:5, 0:2] -> C[2:4, 0:2]
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A[2:4, 0:2] * B[0:2, 2:5] -> C[2:4, 2:5]
    for i in range(2, 4):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A[2:4, 2:5] * B[2:5, 2:5] -> C[2:4, 2:5]
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
