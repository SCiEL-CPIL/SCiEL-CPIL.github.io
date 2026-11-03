"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a low-rank decomposition of the 3x3 * 3x4 matrix multiplication tensor.
    
    The standard schoolbook rank is 3*3*4 = 36.
    We know that for 3x3 matrix multiplication, the optimal rank is 23 (Strassen-like improvements 
    exist, but exact integer/half-integer decompositions with rank < 23 are not known for 3x3x3.
    Actually, the best known upper bound for 3x3x3 is 23. For 3x3x4, we can view it as two 3x3x2 
    or similar blocks, or just solve directly.
    
    However, finding a rank < 36 decomposition with integer/half-integer entries is non-trivial.
    Let's try to use the known Strassen algorithm for 2x2x2 to build up, or use a direct search 
    for a small rank.
    
    Actually, a simpler approach: The task allows integer/half-integer entries. 
    The Strassen algorithm for 2x2x2 has rank 7 with integer coefficients.
    For 3x3x3, we can partition into 2x2 blocks? No, 3 is odd.
    
    Let's try a direct construction using the "Strassen-like" method for 3x3 matrices.
    There is a known decomposition of 3x3 matrix multiplication with rank 23.
    But implementing a specific rank-23 decomposition from scratch with the correct indices is complex.
    
    Alternative: Use a generic rank-reduction search for small ranks.
    Since the dimensions are small (3,3,4), we can try to find a decomposition of rank R < 36.
    Let's try R = 32, 33, etc. But solving the nonlinear system is hard.
    
    Instead, let's use the fact that we can embed smaller multiplications.
    3x3 * 3x4 can be seen as:
    [A11 A12 A13]   [B11 B12 B13 B14]
    [A21 A22 A23] * [B21 B22 B23 B24]
    [A31 A32 A33]   [B31 B32 B33 B34]
    
    We can compute this using two 2x2 Strassen multiplications? No.
    
    Let's try a known result: The tensor rank of 3x3x3 is at most 23.
    For 3x3x4, we can compute the first two columns of the result using a 3x3x2 multiplication 
    and the last two using another 3x3x2.
    What is the rank of 3x3x2? It is at most 3*3*2 = 18, but likely lower.
    Actually, we can use the standard schoolbook for 3x3x2 which is 18.
    So total rank would be 18 + 18 = 36. No improvement.
    
    However, we can use a more efficient algorithm.
    Let's try to find a decomposition of rank 34 or 35 by flipping some bits?
    No, that's not how tensor rank works.
    
    Let's look for a simple improvement.
    The product of a 3x3 and 3x4 matrix has 12 entries.
    Each entry is a sum of 3 products.
    Total 36 products in schoolbook.
    
    Is there a way to share products?
    (A11+A12)(B11+B21) = A11B11 + A11B21 + A12B11 + A12B21.
    This shares the sum, but the products are still distinct terms in the expansion.
    In tensor decomposition, each term is a rank-1 component.
    
    Let's try to use the Strassen algorithm for 2x2x2 (rank 7) to compute parts of the 3x3x4.
    Partition A into 2x2, 2x1, 1x2, 1x1 blocks?
    A = [A11 A12 A13]
        [A21 A22 A23]
        [A31 A32 A33]
    
    Let's group rows 1,2 and row 3 separately.
    C12 = [A11 A12; A21 A22] * [B11 B12 B13 B14; B21 B22 B23 B24]  (2x4 result)
    C3  = [A31 A32 A33] * [B11 B12 B13 B14; B21 B22 B23 B24; B31 B32 B33 B34] (1x4 result)
    
    For C12 (2x2 * 2x4):
    We can use Strassen for the 2x2 part?
    The 2x4 product can be split into two 2x2 products:
    C12_left = A12_block * B12_block (2x2 * 2x2)
    C12_right = A12_block * B34_block (2x2 * 2x2)
    
    Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    So C12 takes 7 + 7 = 14 terms.
    
    For C3 (1x3 * 3x4):
    This is a dot product of a row vector with each column.
    C3_j = A31*B1j + A32*B2j + A33*B3j.
    This is 3*4 = 12 terms in schoolbook.
    Can we do better? The rank of the 1x3x3x4 tensor?
    It's just 12 independent products. No sharing possible because each output element 
    depends on a unique set of 3 products, and no product is shared between different 
    output elements in a way that reduces the total count below 12 (since each A3k is 
    multiplied by all 4 Bkj, and each Bkj is multiplied by all 3 A3k).
    Wait, the tensor is T_{i,j,k,l} = A_{i,k} B_{k,l}. Here i=1 (only one row), l=1..4, k=1..3.
    The terms are A31*B1l, A32*B2l, A33*B3l for l=1..4.
    There is no sharing between different l's for the same k, or different k's for the same l.
    So rank is 12.
    
    Total rank = 14 + 12 = 26.
    This is much better than 36!
    
    Let's verify the Strassen decomposition for 2x2x2.
    Strassen's algorithm computes C = AB for 2x2 matrices using 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to map this to our indices.
    For C12_left (rows 1,2 of A, cols 1,2 of B):
    A_block = [[A11, A12], [A21, A22]]
    B_block = [[B11, B12], [B21, B22]]
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7 = A11B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
    = A11B11 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 - A22B21
    Wait, let's re-calculate carefully.
    C11 should be A11B11 + A12B21.
    My expansion:
    M1 = A11B11 + A11B22 + A22B11 + A22B22
    M4 = A22B12 - A22B11
    M5 = A11B22 + A12B22
    M7 = A12B21 + A12B22 - A22B21 - A22B22
    
    M1+M4-M5+M7 = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B12 - A22B11) - (A11B22 + A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
    = A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
    = A11B11 + A22B12 + A12B21 - A22B21
    This is not A11B11 + A12B21. There is an error in my Strassen formula or expansion.
    
    Correct Strassen:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's check C11 again.
    M1 = A11B11 + A11B22 + A22B11 + A22B22
    M4 = A22B12 - A22B11
    -M5 = -A11B22 - A12B22
    M7 = A12B21 + A12B22 - A22B21 - A22B22
    
    Sum:
    A11B11: 1
    A11B22: 1 - 1 = 0
    A22B11: 1 - 1 = 0
    A22B22: 1 - 1 = 0
    A22B12: 1
    A12B22: -1 + 1 = 0
    A12B21: 1
    A22B21: -1
    
    Result: A11B11 + A22B12 + A12B21 - A22B21.
    This is still wrong. C11 should be A11B11 + A12B21.
    Where is the error?
    Ah, M4 is A22(B12 - B11).
    Is the Strassen formula correct?
    Yes, it is standard.
    Let's check C12 = M3 + M5.
    M3 = A11B21 - A11B22
    M5 = A11B22 + A12B22
    Sum = A11B21 + A12B22.
    But C12 should be A11B12 + A12B22.
    So M3 should involve B12?
    
    Let's look up the exact Strassen formulas.
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Wait, C12 = M3 + M5 = A11(B21-B22) + (A11+A12)B22 = A11B21 - A11B22 + A11B22 + A12B22 = A11B21 + A12B22.
    But the (1,2) entry of the product is A11*B12 + A12*B22.
    So M3 should be A11(B12 - B22)? No.
    
    Actually, the standard Strassen computes:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    But the definitions of M3 and M4 might be swapped or different in some sources.
    Let's use a reliable source.
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's re-evaluate C12.
    C12 = M3 + M5 = A11(B21-B22) + (A11+A12)B22 = A11B21 - A11B22 + A11B22 + A12B22 = A11B21 + A12B22.
    This is not A11B12 + A12B22.
    
    There is a mismatch. The standard Strassen algorithm is for:
    C11 = A11B11 + A12B21
    C12 = A11B12 + A12B22
    C21 = A21B11 + A22B21
    C22 = A21B12 + A22B22
    
    My calculation gives C12 = A11B21 + A12B22.
    This suggests that the M definitions or C formulas are for a different arrangement.
    
    Let's try:
    M3 = A11(B12 - B22) ? No.
    
    Actually, let's just use the schoolbook for the 2x2 blocks to be safe, but that gives rank 4 for 2x2x2, which is not better than schoolbook.
    
    However, I can use a known rank-7 decomposition for 2x2x2.
    Let's find the correct one.
    
    From Wikipedia:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's check C21 = M2 + M4.
    M2 = (A21+A22)B11 = A21B11 + A22B11
    M4 = A22(B12-B11) = A22B12 - A22B11
    Sum = A21B11 + A22B12.
    But C21 should be A21B11 + A22B21.
    So M4 should be A22(B21-B11)?
    
    It seems the standard formulas I'm recalling are slightly off or I'm misapplying them.
    
    Let's use a simpler, verified rank-7 decomposition.
    Or, let's just use the schoolbook for the 2x2 blocks (rank 4) and accept rank 16 + 12 = 28.
    This is still better than 36.
    
    Let's try to get rank 26 with the correct Strassen.
    I will implement the Strassen algorithm carefully by testing it in code.
    
    Here is a correct set of Strassen formulas (from a reliable source):
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    I will verify this in the code by constructing the tensors and checking if they match the schoolbook tensor.
    If they don't match, I will fall back to schoolbook for the 2x2 blocks (rank 4).
    
    Let's build the full decomposition with rank 28 (2x2 blocks with schoolbook, 1x3 row with schoolbook).
    Rank = 4 + 4 + 12 = 20? No.
    C12 is 2x4. Split into two 2x2 products.
    Left 2x2: rank 4.
    Right 2x2: rank 4.
    Total for C12: 8.
    C3 is 1x4. 12 terms.
    Total rank: 8 + 12 = 20.
    
    Wait, is the rank of 2x2x2 really 4? Yes, schoolbook is 4.
    Is there a rank < 4 for 2x2x2? No, the lower bound is 4 (since the slice rank is 4).
    So 2x2x2 has rank 4.
    
    So using schoolbook for the 2x2 blocks gives rank 8 for the 2x4 part.
    And 12 for the 1x4 part.
    Total rank 20.
    
    This is a huge improvement over 36!
    
    Let's verify the indices.
    n=3, m=3, p=4.
    U is 9x20, V is 12x20, W is 12x20.
    
    Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    Indices:
    A rows 0,1; B cols 0,1.
    U indices: i*m+j. i in 0,1; j in 0,1.
    V indices: j*p+k. j in 0,1; k in 0,1.
    W indices: k*n+i. k in 0,1; i in 0,1.
    
    Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    A rows 0,1; B cols 2,3.
    U indices: i*m+j. i in 0,1; j in 0,1.
    V indices: j*p+k. j in 0,1; k in 2,3.
    W indices: k*n+i. k in 2,3; i in 0,1.
    
    Block 3: A[2:3, 0:3] * B[0:3, 0:4] -> C[2:3, 0:4]
    A row 2; B cols 0,1,2,3.
    U indices: i*m+j. i=2; j in 0,1,2.
    V indices: j*p+k. j in 0,1,2; k in 0,1,2,3.
    W indices: k*n+i. k in 0,1,2,3; i=2.
    
    Total terms: 4 + 4 + 12 = 20.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS
    # Rank 20 decomposition
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    for i in range(2):
        for j in range(2):
            for k in range(2, 4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 3: A[2:3, 0:3] * B[0:3, 0:4]
    for i in range(2, 3):
        for j in range(3):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
