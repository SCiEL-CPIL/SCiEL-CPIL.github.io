"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) lists of lists, rank n*m*p."""
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
    return U, V, W


def _combine(U1, V1, W1, U2, V2, W2, n, m, p):
    """Concatenate two decompositions horizontally (block diag of factors)."""
    r1 = len(W1[0])
    r2 = len(W2[0])
    r = r1 + r2
    U = []
    for row in range(n * m):
        U.append(U1[row] + U2[row])
    V = []
    for row in range(m * p):
        V.append(V1[row] + V2[row])
    W = []
    for row in range(p * n):
        W.append(W1[row] + W2[row])
    return U, V, W


def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    A is 2x2 with rows [a b; c d] flattened row-major: (a,b,c,d)
    B is 2x2 with rows [e f; g h] flattened row-major: (e,f,g,h)
    C = A*B has rows [a*e+b*g, a*f+b*h; c*e+d*g, c*f+d*h]
    W index is k*n + i = k*2 + i (since n=2, p=2).
    W[0] -> i=0,k=0: C[0][0] = a*e + b*g
    W[1] -> i=0,k=1: C[0][1] = a*f + b*h
    W[2] -> i=1,k=0: C[1][0] = c*e + d*g
    W[3] -> i=1,k=1: C[1][1] = c*f + d*h
    """
    # Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0][0] = M1 - M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 - M3 + M6
    #
    # A index: a=0, b=1, c=2, d=3
    # B index: e=0, f=1, g=2, h=3

    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a: M1, M3, M5, M6
        [0, 0, -1, 0, 1, 0, 1],   # b: M3, M5, M7
        [0, 1, 0, 0, 0, 1, 0],   # c: M2, M6
        [1, 1, 0, 1, 0, 0, -1]    # d: M1, M2, M4, M7
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],   # e: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0],   # f: M3, M6
        [0, 0, 0, 1, 0, 0, 1],   # g: M4, M7
        [1, 0, 0, 0, 1, 0, 1]    # h: M1, M5, M7
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],   # C[0][0] = M1-M4-M5+M7
        [0, 0, 1, 0, 1, 0, 0],     # C[0][1] = M3+M5
        [0, 1, 0, 1, 0, 0, 0],     # C[1][0] = M2+M4
        [1, -1, -1, 0, 0, 1, 0]    # C[1][1] = M1-M2-M3+M6
    ]
    return U, V, W


def _strassen_3x3():
    """Strassen's 23-product algorithm for 3x3x3.
    This is complex; we'll use a known set of factors.
    A is 3x3 flattened row-major: indices 0..8
    B is 3x3 flattened row-major: indices 0..8
    C is 3x3, W index = k*3 + i (p=3, n=3)
    
    For simplicity and correctness, I'll use the 23-product Strassen 3x3.
    Reference: Strassen 1969, or standard implementations.
    
    Actually, let me use a simpler approach: I'll implement the 3x3 using
    7 smaller multiplications where possible, but the standard Strassen 3x3
    uses 23 products. Let me hardcode a known correct set.
    
    Given the complexity and risk of error, let me instead build the 3x3
    decomposition by combining 2x2 Strassen blocks with additional terms.
    
    A 3x3 * 3x3 can be seen as:
    [ A11 A12 | A13   ] [ B11 B12 | B13   ]
    [ A21 A22 | A23   ] [ B21 B22 | B23   ]
    [ ------| ----   ] [ ------| ----   ]
       2x2   2x1        2x2   2x1
    
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2] + A[0:2,0:2]... 
    
    This is getting complicated. Let me just use schoolbook for the 3x3 blocks
    and rely on the 2x2 Strassen where it fits.
    
    Alternative: For 3x3x3, known rank is 23 (Strassen) or lower (Baur-Strassen 
    shows at most 23, and recent results show it's 23 or less).
    
    Given time constraints and correctness, let me use a hybrid:
    - Use Strassen 2x2 for the top-left 2x2 block
    - Handle remaining 2x3 and 3x2 interactions with schoolbook-like terms
    
    Actually, let me just do a careful block decomposition.
    
    For simplicity and guaranteed correctness, I'll use the following strategy:
    The rank of 3x3x3 is at most 27 (schoolbook) and known to be <= 23.
    Since I can't easily verify a 23-factor set, let me just use schoolbook 
    for 3x3 to ensure correctness, and focus on getting the 4x5x5 right.
    """
    # Use schoolbook for 3x3 to ensure correctness
    return _schoolbook(3, 3, 3)


def _try_improved_3x3():
    """Try to build a better 3x3 decomposition.
    Returns None if no improvement over schoolbook (27) is found quickly.
    """
    # Strassen 3x3 uses 23 products. Let me hardcode a known set.
    # Reference: https://en.wikipedia.org/wiki/Strassen%27s_algorithm#3x3_matrices
    # 
    # The 23 products are based on partitioning 3x3 into 2x2 and 1x1 blocks.
    # A = [A1 A2; A3 A4] where A1 is 2x2, A2 is 2x1, A3 is 1x2, A4 is 1x1
    # Same for B.
    #
    # C = A*B
    # C11 (2x2) = A1*B1 + A2*B3
    # C12 (2x1) = A1*B2 + A2*B4  
    # C21 (1x2) = A3*B1 + A4*B3
    # C22 (1x1) = A3*B2 + A4*B4
    #
    # Using Strassen on A1*B1 (7 products), plus cross terms:
    # A2*B3 (2x1 * 1x2 = 2x2, rank 4 schoolbook or less)
    # A1*B2 (2x2 * 2x1, rank 4)
    # A3*B1 (1x2 * 2x2, rank 4)
    # A4*B4 (1x1 * 1x1, rank 1)
    # A2*B4 (2x1 * 1x1, rank 2)
    # A3*B2 (1x2 * 2x1, rank 2)
    # A4*B3 (1x1 * 1x2, rank 2)
    # A1... wait this is getting messy.
    #
    # Total: 7 + 4 + 4 + 4 + 1 + 2 + 2 + 2 = 26. Not better than 27.
    # 
    # A more careful Strassen 3x3 achieves 23. Let me just use schoolbook.
    return None


def _construct_4x4x4():
    """Construct 4x4x4 decomposition.
    Split 4x4 into 2x2 + 2x2 blocks:
    A = [A1 A2; A3 A4] where each is 2x2
    B = [B1 B2; B3 B4]
    
    C = A*B
    C11 (2x2) = A1*B1 + A2*B3
    C12 (2x2) = A1*B2 + A2*B4
    C21 (2x2) = A3*B1 + A4*B3
    C22 (2x2) = A3*B2 + A4*B4
    
    Each 2x2*2x2 uses Strassen (7 products)
    Cross terms: A2*B3, A2*B4, A3*B1, A4*B3, A3*B2, A4*B4
    But some of these ARE the 2x2*2x2 multiplications!
    
    Actually:
    C11 = A1*B1 + A2*B3
    C12 = A1*B2 + A2*B4
    C21 = A3*B1 + A4*B3
    C22 = A3*B2 + A4*B4
    
    The 8 distinct 2x2*2x2 products needed:
    A1*B1, A1*B2, A1*B3, A1*B4, A2*B3, A2*B4, A3*B1, A3*B2, A4*B3, A4*B4
    Wait, that's 10. Let me recount.
    
    Products of the form A_XX * B_YZ where XX, YZ ∈ {1,2,3,4}:
    For C11: A1*B1, A2*B3
    For C12: A1*B2, A2*B4
    For C21: A3*B1, A4*B3
    For C22: A3*B2, A4*B4
    
    So 8 distinct 2x2*2x2 products. Each costs 7 (Strassen). Total: 56.
    
    But we can do better. The 4x4x4 tensor has known rank 49 (Baur-Strassen / 
    Stott 1971). Let me try to construct it.
    
    Alternative: Use the fact that 4x4x4 can be decomposed more cleverly.
    
    Given time, let me use a simpler block approach:
    Treat 4x4 as (2+2)x(2+2). Use Strassen for the "big" Strassen on 2x2 blocks.
    This gives rank 7 * 4 = 28? No, that's not right either.
    
    Let me just use schoolbook for 4x4x4 (rank 64) or try a simple improvement.
    """
    # Schoolbook: rank 64
    # Try: split into 4 2x2*2x2 multiplications (Strassen, 7 each) 
    #      but that only handles the diagonal blocks of C.
    # For the off-diagonal blocks, we need more.
    #
    # Simplest correct approach: schoolbook with rank 64.
    # But let's try a block Strassen:
    # View A as 2x2 blocks of 2x2, same for B.
    # Then C is 2x2 blocks of 2x2.
    # C_ij = sum_k A_ik * B_kj
    # Use Strassen on the 2x2 "super-matrices" where each entry is 2x2.
    # Strassen: 7 "super-products", each is 2x2*2x2 which is rank 7 (Strassen).
    # Total: 7 * 7 = 49. This matches the known rank!
    
    # So: 4x4x4 = Strassen(2x2 super, each entry 2x2) = 49 rank.
    #
    # Let me implement this by composing Strassen at two levels.
    
    strassen_2x2_U, strassen_2x2_V, strassen_2x2_W = _strassen_2x2()
    
    # Level 1: 2x2 super-matrix Strassen
    # Super A has entries A11, A12, A21, A22 (each 2x2)
    # Super B has entries B11, B12, B21, B22 (each 2x2)
    # Super C = A*B via 7 super-products
    
    # Each super-product is a sum of products of 2x2 matrices.
    # Strassen super-products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    
    # Each "super-product" like (A11+A22)(B11+B22) is a 2x2*2x2 product
    # of the SUMS. We can factor this as:
    # (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # This is 4 separate 2x2*2x2 products, each rank 7. So 28 for M1 alone.
    # That's too much.
    #
    # Better: use Strassen directly on each super-product.
    # (A11+A22)(B11+B22) - we treat this as a single 2x2*2x2 multiplication
    # where the "matrices" are A' = A11+A22 and B' = B11+B22.
    # Each of these is 2x2. So this is one 2x2*2x2 = rank 7.
    # 
    # So total for 4x4x4: 7 super-products * 7 (Strassen 2x2) = 49.
    
    # Let me build this carefully.
    # A is 4x4, indices 0-15. Row-major.
    # A11 = rows 0-1, cols 0-1: indices (0,1,4,5)
    # A12 = rows 0-1, cols 2-3: indices (2,3,6,7)
    # A21 = rows 2-3, cols 0-1: indices (8,9,12,13)
    # A22 = rows 2-3, cols 2-3: indices (10,11,14,15)
    #
    # B11 = rows 0-1, cols 0-1: indices (0,1,4,5)
    # B12 = rows 0-1, cols 2-3: indices (2,3,6,7)
    # B21 = rows 2-3, cols 0-1: indices (8,9,12,13)
    # B22 = rows 2-3, cols 2-3: indices (10,11,14,15)
    #
    # C indices: C[i][j], i=0..3, j=0..3. W index = k*4 + i where k is col of B, i is row of A.
    # C11 = rows 0-1, cols 0-1: W indices k*4+i for i in {0,1}, k in {0,1}
    #   k=0,i=0: 0; k=1,i=0: 1; k=0,i=1: 4; k=1,i=1: 5
    # C12 = rows 0-1, cols 2-3: W indices
    #   k=2,i=0: 2; k=3,i=0: 3; k=2,i=1: 6; k=3,i=1: 7
    # C21 = rows 2-3, cols 0-1: W indices
    #   k=0,i=2: 8; k=1,i=2: 9; k=0,i=3: 12; k=1,i=3: 13
    # C22 = rows 2-3, cols 2-3: W indices
    #   k=2,i=2: 10; k=3,i=2: 11; k=2,i=3: 14; k=3,i=3: 15
    
    # I'll build the 4x4x4 by direct enumeration using the 2-level Strassen.
    # This is complex to hardcode. Let me use a simpler approach:
    # Just use schoolbook for 4x4x4 (rank 64) since constructing the 49-rank
    # decomposition correctly is error-prone and time-consuming.
    
    return _schoolbook(4, 4, 4)


def _construct_1x3x5():
    """1x3 matrix times 3x5 matrix = 1x5. Rank 3 (schoolbook is optimal)."""
    n, m, p = 1, 3, 5
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def _construct_4x2x1():
    """4x2 matrix times 2x1 matrix = 4x1. Rank 2 (schoolbook is optimal)."""
    n, m, p = 4, 2, 1
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def _construct_1x3x1():
    """1x3 times 3x1 = 1x1. Rank 3 (schoolbook is optimal)."""
    n, m, p = 1, 3, 1
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def _construct_2x3x2():
    """2x3 times 3x2 = 2x2. Schoolbook rank 12, but we can do better.
    A = [a b c; d e f], B = [g h; i j; k l]
    C[0][0] = ag+bh+ck, C[0][1] = ah+bj+cl
    C[1][0] = dg+ei+fk, C[1][1] = dh+ej+fl
    
    Using Strassen-like: treat as (2+0)x(3) x (3)x(2+0).
    Or split A into (1+1)x3, B into 3x(1+1).
    
    Simplest: schoolbook rank 12. But known rank of 2x3x2 is at most 11 or so.
    Let me just use schoolbook for correctness.
    """
    n, m, p = 2, 3, 2
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def _construct_4x5x5():
    """Build 4x5x5 by decomposing into blocks.
    
    A is 4x5, B is 5x5, C is 4x5.
    Split columns of A: A = [A_left | A_right] where A_left is 4x2, A_right is 4x3
    Split rows of B correspondingly: B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 3x5
    
    C = A*B = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 4x2 * 2x5 = 4x5. Schoolbook rank 40.
    A_right * B_bottom: 4x3 * 3x5 = 4x5. Schoolbook rank 60.
    Total: 100 (schoolbook).
    
    Better: split differently.
    A is 4x5, split into 2x5 + 2x5 (rows).
    A = [A_top; A_bottom] where A_top is 2x5, A_bottom is 2x5.
    B is 5x5, keep as is.
    
    C = A_top * B + A_bottom * B
    A_top * B: 2x5 * 5x5 = 2x5. Schoolbook rank 50.
    A_bottom * B: same, rank 50.
    Total: 100. Still same.
    
    Let me use Strassen where it helps.
    Split A as 4x5 = (2+2)x5, B as 5x5 = 5x(2+3).
    
    A = [A1; A2] where A1, A2 are 2x5.
    B = [B1 | B2] where B1 is 5x2, B2 is 5x3.
    
    C = [A1*B1 + A1*B2 | ...] wait, C is 4x5.
    C_top (2x5) = A1 * B = A1*B1 + A1*B2  where B1 is 5x2, B2 is 5x3? No.
    
    Let me redo. A is 4x5, B is 5x5.
    A rows: 0,1,2,3. B rows: 0,1,2,3,4.
    
    C[i][k] = sum_j A[i][j] * B[j][k]
    
    Block by A rows (0-1) and (2-3), and B rows (0-1) and (2-4):
    
    This is getting messy. Let me just use a clean block decomposition.
    
    Split A into two 2x5 row blocks: A = [A1; A2], A1, A2 in R^{2x5}
    Split B into two 5x2/5x3 col blocks? No, B is 5x5.
    
    C_top (2x5) = A1 * B
    C_bottom (2x5) = A2 * B
    
    For A1*B (2x5 * 5x5): split A1 into 2x2 + 2x3, B into (2+3)x5.
    A1 = [A1a | A1b] where A1a is 2x2, A1b is 2x3.
    B = [B1; B2] where B1 is 2x5, B2 is 3x5.
    A1*B = A1a*B1 + A1b*B2
    A1a*B1: 2x2 * 2x5. Schoolbook rank 20. Can we do better?
      2x2 * 2x5 = 2x5. Use Strassen for 2x2*2x2 part... but B1 is 2x5.
      Split B1 into B1_left (2x2) + B1_right (2x3).
      A1a*B1 = A1a*B1_left + A1a*B1_right
      A1a*B1_left: 2x2*2x2, Strassen rank 7.
      A1a*B1_right: 2x2*2x3, schoolbook rank 12. Or better?
      2x2 * 2x3: A1a is 2x2, B1_right is 2x3.
      C = A1a * B1_right, 2x3 output.
      Schoolbook: 12 products.
      Can use: for each column of B1_right (2x1), multiply 2x2 * 2x1 = 2x1, rank 4.
      3 columns: rank 12. No improvement.
      
      So A1a*B1: 7 + 12 = 19.
      A1b*B2: 2x3 * 3x5, schoolbook rank 30.
      
      A1*B total: 19 + 30 = 49.
    Similarly A2*B: 49.
    Total: 98. Better than 100 but not great.
    
    Let me just go with a simpler but still improved construction.
    
    Actually, the simplest correct improvement: 
    Use the 2x2 Strassen for 2x2x2 subproblems wherever possible.
    
    For 4x5x5, let me use schoolbook (rank 100) since constructing a better
    decomposition correctly within time limits is risky.
    """
    return _schoolbook(4, 5, 5)


def construct():
    """Construct the rank-minimal (as far as we can determine) exact tensor
    decomposition for multiplying a 4x5 matrix by a 5x5 matrix."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(W[0])  # 100
    
    # Try block-based improvements
    # Decompose 4x5 * 5x5 into sub-problems using Strassen where applicable
    
    # Strategy: Split A (4x5) into 2x5 + 2x5 (rows 0-1, 2-3)
    # Split B (5x5) into 2x5 + 3x5 (rows 0-1, 2-4)
    #
    # C_top (2x5) = A_top * B
    # C_bot (2x5) = A_bot * B
    #
    # For A_top * B (2x5 * 5x5):
    #   Split B columns? No, B is 5x5, we multiply 2x5 by 5x5.
    #   Split B rows into 2 + 3: B = [B1; B2]
    #   A_top * B = A_top * [B1; B2] 
    #   But A_top is 2x5, B1 is 2x5, B2 is 3x5.
    #   A_top * B1 is not valid (2x5 * 2x5 = 2x5? No, dims don't match for *).
    #   
    # I keep making dimensional errors. Let me be very careful.
    #
    # A: 4x5. Rows indexed 0..3, cols 0..4.
    # B: 5x5. Rows indexed 0..4, cols 0..4.
    # C = A*B: 4x5. C[i][k] = sum_{j=0}^{4} A[i][j] B[j][k].
    #
    # Split the summation j into j=0,1 and j=2,3,4:
    # C[i][k] = (A[i][0]B[0][k] + A[i][1]B[1][k]) + (A[i][2]B[2][k] + A[i][3]B[3][k] + A[i][4]B[4][k])
    #
    # Define A1[i] = [A[i][0], A[i][1]] for i=0..3 (this is A[:, 0:2], a 4x2 matrix)
    # Define A2[i] = [A[i][2], A[i][3], A[i][4]] for i=0..3 (this is A[:, 2:5], a 4x3 matrix)
    # Define B1[k] = [B[0][k], B[1][k]] for k=0..4 (this is B[0:2, :], a 2x5 matrix)
    # Define B2[k] = [B[2][k], B[3][k], B[4][k]] for k=0..4 (this is B[2:5, :], a 3x5 matrix)
    #
    # Then C = A1 * B1 + A2 * B2  where:
    # A1: 4x2, B1: 2x5, product: 4x5
    # A2: 4x3, B2: 3x5, product: 4x5
    
    # A1*B1: 4x2 * 2x5
    #   Schoolbook: 4*2*5 = 40.
    #   Better: Split A1 into 2x2 + 2x2 (rows 0-1, 2-3), B1 into 2x2 + 2x3 (cols 0-1, 2-4)
    #   A1 = [A1_top; A1_bot] (each 2x2)
    #   B1 = [B1_left | B1_right] (B1_left is 2x2, B1_right is 2x3)
    #   A1*B1 = A1_top*B1_left + A1_top*B1_right + A1_bot*B1_left + A1_bot*B1_right
    #   A1_top*B1_left: 2x2*2x2, Strassen rank 7
    #   A1_top*B1_right: 2x2*2x3, schoolbook rank 12
    #   A1_bot*B1_left: 2x2*2x2, Strassen rank 7
    #   A1_bot*B1_right: 2x2*2x3, schoolbook rank 12
    #   Total: 7+12+7+12 = 38
    
    # A2*B2: 4x3 * 3x5
    #   Schoolbook: 4*3*5 = 60
    #   Split A2 into 2x3 + 2x3 (rows 0-1, 2-3)
    #   A2 = [A2_top; A2_bot] (each 2x3)
    #   A2*B2 = A2_top*B2 + A2_bot*B2
    #   A2_top*B2: 2x3 * 3x5, schoolbook rank 30
    #   A2_bot*B2: 2x3 * 3x5, schoolbook rank 30
    #   Total: 60
    
    # Grand total: 38 + 60 = 98. Slight improvement over 100.
    
    # Let me build this explicitly.
    
    if time.time() - t0 > _DEADLINE_SECONDS - 20:
        return {"U": U, "V": V, "W": W}
    
    # Build A1*B1 part (4x2 * 2x5)
    # A1 indices in original A (4x5): rows i=0..3, cols j=0..1
    #   A1[i][j] = A[i * 5 + j] for i=0..3, j=0..1
    #   In U matrix (shape n*m = 4*5 = 20 rows), row index = i*5+j
    #   j=0: rows 0, 5, 10, 15
    #   j=1: rows 1, 6, 11, 16
    
    # B1 indices in original B (5x5): rows j=0..1, cols k=0..4
    #   B1[j][k] = B[j * 5 + k] for j=0..1, k=0..4
    #   In V matrix (shape m*p = 5*5 = 25 rows), row index = j*5+k
    #   j=0: rows 0,1,2,3,4
    #   j=1: rows 5,6,7,8,9
    
    # C part for A1*B1: C[i][k] gets contributions, W index = k*4 + i
    # i=0..3, k=0..4
    
    # Sub-problems for A1*B1:
    # A1_top*B1_left: A1 rows 0-1, B1 cols 0-1
    #   A1_top: A[0,0], A[0,1], A[1,0], A[1,1] -> U rows 0,1,5,6
    #   B1_left: B1 rows 0-1, cols 0-1 -> V rows 0,1,5,6
    #   Output: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,1,4,5
    #   This is 2x2*2x2, use Strassen.
    
    # A1_top*B1_right: A1 rows 0-1, B1 cols 2-4
    #   A1_top: U rows 0,1,5,6
    #   B1_right: B1 rows 0-1, cols 2-4 -> V rows 2,3,4,7,8,9
    #   Output: C[0,2], C[0,3], C[0,4], C[1,2], C[1,3], C[1,4]
    #   W indices: k*4+i for i in {0,1}, k in {2,3,4}
    #   k=2,i=0:8; k=3,i=0:9; k=4,i=0:10; k=2,i=1:12; k=3,i=1:13; k=4,i=1:14
    #   This is 2x2 * 2x3, schoolbook rank 12.
    
    # A1_bot*B1_left: A1 rows 2-3, B1 cols 0-1
    #   A1_bot: A[2,0], A[2,1], A[3,0], A[3,1] -> U rows 10,11,15,16
    #   B1_left: V rows 0,1,5,6
    #   Output: C[2,0], C[2,1], C[3,0], C[3,1] -> W indices 8,9,12,13
    #   2x2*2x2, Strassen.
    
    # A1_bot*B1_right: A1 rows 2-3, B1 cols 2-4
    #   A1_bot: U rows 10,11,15,16
    #   B1_right: V rows 2,3,4,7,8,9
    #   Output: C[2,2]..C[2,4], C[3,2]..C[3,4]
    #   W indices: k=2,i=2:10; k=3,i=2:11; k=4,i=2:12; k=2,i=3:14; k=3,i=3:15; k=4,i=3:16
    #   2x2 * 2x3, schoolbook rank 12.
    
    # Now A2*B2 (4x3 * 3x5):
    # A2: A[i][j] for i=0..3, j=2..4 -> U rows i*5+j for j=2,3,4
    #   i=0: rows 2,3,4; i=1: rows 7,8,9; i=2: rows 12,13,14; i=3: rows 17,18,19
    # B2: B[j][k] for j=2..4, k=0..4 -> V rows j*5+k for j=2,3,4
    #   j=2: rows 10,11,12,13,14; j=3: rows 15,16,17,18,19; j=4: rows 20,21,22,23,24
    # Output: all C[i][k] for i=0..3, k=0..4 -> W indices 0..19
    # This overlaps with A1*B1 outputs! We need to add them together.
    
    # This block decomposition with overlapping outputs means I can't simply
    # concatenate. I need to be more careful.
    
    # Given the complexity and risk of errors, let me just return schoolbook.
    # The current score is 1.0 (rank 100), and improving to ~98 gives score ~1.02.
    # To make a meaningful jump, I'd need rank well below 100.
    
    # Let me try one more thing: a simple Strassen-based 4x4x4 subproblem
    # combined with simple terms for the extra column/row.
    
    # 4x5 * 5x5 = (4x4 * 4x4) + (4x1 * 4x1 terms) + (4x4 * 1x5) + (4x1 * 1x5) + (4x1 * 1x5)
    # This is getting too complex to do correctly in limited time.
    
    # Final decision: return schoolbook rank 100 for correctness.
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
