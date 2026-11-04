"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import time
_START = time.time()
_DEADLINE = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _schoolbook():
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

def construct():
    """
    Use Strassen block decomposition to reduce rank.
    
    For 3x5 * 5x5:
    - Split A (3x5) into blocks: top row (1x5) and bottom (2x5)
    - Split B (5x5) into columns: left (5x3) and right (5x2) -- no, let's think differently
    
    Better approach: treat the 3x5 * 5x5 as blocks.
    Let's partition:
      A is 3x5, B is 5x5, C = A*B is 3x5
    
    We can use the fact that 3 = 1+2 and 5 = 3+2 for A's rows and B's cols? 
    Actually, let's use a cleaner block structure.
    
    Partition A (3x5) as:
      A = [A11 (1x5)]
          [A21 (2x5)]
    
    Partition B (5x5) as:
      B = [B11 (5x3)  B12 (5x2)]
    
    Then C = A*B = [C11 (1x3)  C12 (1x2)]
                   [C21 (2x3)  C22 (2x2)]
    
    C11 = A11 * B11  (1x3)
    C12 = A11 * B12  (1x2)
    C21 = A21 * B11  (2x3)
    C22 = A21 * B12  (2x2)
    
    For the 2x2 blocks (A21 is 2x5, B12 is 5x2, etc.), we can't directly use Strassen 
    because the inner dimension is 5, not 2.
    
    Let me reconsider. Strassen works for n x n * n x n. Here we have non-square blocks.
    
    Alternative: Use the 7-term Strassen for the 2x2 * 2x2 part, and handle the rest.
    
    Let's partition more carefully:
    A (3x5): rows 0, 1, 2
    B (5x5): cols 0, 1, 2, 3, 4
    
    Split A into A_top (1x5) and A_bot (2x5)
    Split B into B_left (5x3) and B_right (5x2)
    
    C_top_left = A_top * B_left  (1x3)
    C_top_right = A_top * B_right  (1x2)
    C_bot_left = A_bot * B_left  (2x3)
    C_bot_right = A_bot * B_right  (2x2)
    
    Now, A_bot is 2x5 and B_left is 5x3. We can further split these.
    
    Actually, let me try a different decomposition. Since 5 = 3 + 2, let's split:
    A (3x5) = [A1 (3x3) | A2 (3x2)]
    B (5x5) = [B1 (5x3)]
               [B2 (5x2)]
    
    C = A*B = [A1*B1 + A2*B2]  (3x3)
    
    Hmm, this is getting complicated. Let me try a simpler approach.
    
    Known result: For m x n * n x p, the rank is at most m*n*p but can be reduced.
    For 3x5 * 5x5, known lower bound is around 30-40.
    
    Let me use a block decomposition with Strassen on 2x2 sub-blocks.
    
    Partition:
    A (3x5): 
      Row block 0: row 0 (1x5)
      Row block 1: rows 1-2 (2x5)
    
    B (5x5):
      Col block 0: cols 0-2 (5x3)
      Col block 1: cols 3-4 (5x2)
    
    C (3x5):
      C[0, 0:3] = A[0,:] * B[:,0:3]  (1x3)
      C[0, 3:5] = A[0,:] * B[:,3:5]  (1x2)
      C[1:3, 0:3] = A[1:3,:] * B[:,0:3]  (2x3)
      C[1:3, 3:5] = A[1:3,:] * B[:,3:5]  (2x2)
    
    The 2x2 block C[1:3, 3:5] = A_bot * B_right where A_bot is 2x5 and B_right is 5x2.
    This is a 2x5 * 5x2 multiplication. We can use Strassen-like methods here.
    
    For 2x5 * 5x2:
    A_bot = [a11 a12 a13 a14 a15]
            [a21 a22 a23 a24 a25]
    B_right = [b11 b12]
               [b21 b22]
               [b31 b32]
               [b41 b42]
               [b51 b52]
    
    C = A_bot * B_right (2x2)
    
    We can decompose this as:
    C = [a11 b11 + a12 b21 + a13 b31 + a14 b41 + a15 b51,  a11 b12 + a12 b22 + a13 b32 + a14 b42 + a15 b52]
        [a21 b11 + a22 b21 + a23 b31 + a24 b41 + a25 b51,  a21 b12 + a22 b22 + a23 b32 + a24 b42 + a25 b52]
    
    This is 4 multiplications of vectors, but we can use Strassen on the 2x2 structure.
    
    Actually, for 2xn * nx2, the rank is known to be 7 (same as 2x2x2 Strassen) for n>=2.
    
    Let me use the known 7-term formula for 2x2 matrix multiplication and extend.
    
    For the full problem, let me just use a combination of schoolbook and Strassen.
    
    Given the complexity, let me start with a reasonable decomposition and verify.
    
    I'll use the following approach:
    - Use schoolbook for most parts
    - Use Strassen (7 terms) for the 2x2 * 2x2 sub-problem
    
    Total rank estimate:
    - C[0, 0:3]: 1x3, needs 3 terms (schoolbook)
    - C[0, 3:5]: 1x2, needs 2 terms (schoolbook)
    - C[1:3, 0:3]: 2x3, needs 6 terms (schoolbook)
    - C[1:3, 3:5]: 2x2, needs 7 terms (Strassen)
    
    Total: 3 + 2 + 6 + 7 = 18? That seems too low. The issue is that these blocks share factors.
    
    Let me be more careful. The tensor decomposition is global, not block-wise.
    
    Given the time constraints and complexity, let me return the schoolbook solution as a baseline,
    which is guaranteed correct.
    """
    return _schoolbook()
# EVOLVE-BLOCK-END
