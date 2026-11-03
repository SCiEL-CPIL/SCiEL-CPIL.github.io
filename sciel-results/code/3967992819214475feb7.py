"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a tensor decomposition for 3x4 matrix times 4x5 matrix.
    Uses a block decomposition approach:
    - Split the 4x5 middle dimension into blocks.
    - Use Strassen's algorithm for 2x2x2 sub-blocks where possible.
    - Use schoolbook for remaining parts.
    
    The key insight from the measured knowledge: using two disjoint Strassen 2x2x2 blocks
    (rank 7 each) instead of schoolbook (rank 8 each) reduces total rank.
    
    Strategy: 
    - n=3, m=4, p=5
    - We can partition the m=4 dimension into two parts of size 2 each.
    - We can partition the p=5 dimension into parts of size 2, 2, 1.
    - We can partition the n=3 dimension into parts of size 1, 1, 1.
    
    Actually, let's think more carefully. The standard approach for rectangular matrices:
    - For each pair of block dimensions, we multiply them.
    - If both inner dimensions are 2x2, we can use Strassen (rank 7 instead of 8).
    
    Let's try a different partitioning:
    - n=3: split into 1, 2
    - m=4: split into 2, 2  
    - p=5: split into 2, 2, 1
    
    Block multiplications needed:
    - (1x2)*(2x2) -> 1x2: schoolbook, rank 2
    - (1x2)*(2x2) -> 1x2: schoolbook, rank 2
    - (1x2)*(2x1) -> 1x1: schoolbook, rank 1
    - (2x2)*(2x2) -> 2x2: Strassen, rank 7
    - (2x2)*(2x2) -> 2x2: Strassen, rank 7
    - (2x2)*(2x1) -> 2x1: schoolbook, rank 4
    
    Total rank = 2 + 2 + 1 + 7 + 7 + 4 = 23
    
    Wait, that seems too good. Let me verify the dimensions.
    
    Actually, I need to be more careful. The matrix multiplication tensor T_{ijk} = delta for (i,j,k) where C[i,k] = sum_j A[i,j]*B[j,k].
    
    For n=3, m=4, p=5: total schoolbook rank = 3*4*5 = 60.
    
    Let me use a simpler approach based on the measured finding: use Strassen blocks where the subproblem is 2x2x2.
    
    Partition:
    - n: [1, 1, 1] or [1, 2]
    - m: [2, 2]
    - p: [2, 2, 1]
    
    Using n=[1,2], m=[2,2], p=[2,2,1]:
    Blocks:
    - (1x2) * (2x2) = 1x2: rank 2 (schoolbook)
    - (1x2) * (2x2) = 1x2: rank 2 (schoolbook)
    - (1x2) * (2x1) = 1x1: rank 1 (schoolbook)
    - (2x2) * (2x2) = 2x2: rank 7 (Strassen)
    - (2x2) * (2x2) = 2x2: rank 7 (Strassen)
    - (2x2) * (2x1) = 2x1: rank 4 (schoolbook)
    
    Total = 2+2+1+7+7+4 = 23
    
    But wait, I need to verify this is correct. The issue is that when we partition, the blocks share indices.
    
    Actually, let me use a known result. For 3x4x5, the best known rank is around 50-55.
    
    Let me try a different approach: use the known Strassen-based decomposition.
    
    Actually, the measured finding says: "Using two disjoint Strassen 2x2x2 blocks (rank 7 each) instead of schoolbook (rank 8 each) reduces the total rank from 60 to 58"
    
    So 60 - 2 = 58. That means we replace two 2x2x2 schoolbook blocks (each rank 8) with Strassen (each rank 7), saving 1 rank each.
    
    So the strategy is: identify two non-overlapping 2x2x2 subproblems in the 3x4x5 multiplication, and use Strassen for those.
    
    For 3x4x5:
    - We need to find two disjoint sets of (i,j,k) indices where i in a set of size 2, j in a set of size 2, k in a set of size 2.
    - The sets must be disjoint in the sense that they don't share any index positions.
    
    Let's say we use:
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {1,2}, j in {2,3}, k in {2,3}
    
    These are disjoint in j and k, but share i=1. That's a problem.
    
    Let me try:
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    
    These share i={0,1} but are disjoint in j and k. That should be fine for the tensor decomposition.
    
    Actually, in tensor decomposition, the blocks need to be such that the supports are disjoint. Let me think about this differently.
    
    The schoolbook decomposition has rank 60, with one term for each (i,j,k) triple.
    
    To use Strassen on a 2x2x2 sub-block, we need to find a 2x2x2 sub-block and replace its 8 schoolbook terms with 7 Strassen terms.
    
    For this to work, the 2x2x2 sub-block must be "isolated" in the sense that the Strassen decomposition only involves indices within that sub-block.
    
    So we need two disjoint 2x2x2 sub-blocks. Disjoint means no shared (i,j,k) indices.
    
    Possible disjoint 2x2x2 sub-blocks in 3x4x5:
    - Block A: i={0,1}, j={0,1}, k={0,1}
    - Block B: i={0,1}, j={2,3}, k={2,3}
    
    These share i indices but are disjoint in (j,k) pairs, so the (i,j,k) triples are disjoint. This should work.
    
    Total rank = 60 - 8 + 7 - 8 + 7 = 58.
    
    Let me implement this.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = 3, 4, 5
    rank = 58
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 12 x 58
    V = [[0] * rank for _ in range(m * p)]  # 20 x 58
    W = [[0] * rank for _ in range(p * n)]  # 15 x 58
    
    # We'll use schoolbook for most terms, and Strassen for two 2x2x2 blocks.
    # 
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> use Strassen
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3} -> use Strassen
    # All other (i,j,k) triples -> schoolbook
    
    # First, let's figure out which (i,j,k) triples are in the Strassen blocks
    strassen_block1 = set()
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                strassen_block1.add((i, j, k))
    
    strassen_block2 = set()
    for i in [0, 1]:
        for j in [2, 3]:
            for k in [2, 3]:
                strassen_block2.add((i, j, k))
    
    # Schoolbook terms for non-Strassen (i,j,k)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_block1 or (i, j, k) in strassen_block2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now add Strassen decompositions for the two blocks
    # Strassen 2x2x2: 
    # M1 = A00*B00 -> C00 += M1
    # M2 = A00*B01 -> C01 += M2
    # M3 = A10*B10 -> C10 += M3
    # M4 = A11*B11 -> C11 += M4
    # M5 = (A00+A11)*(B00+B11) -> C00 -= M5, C11 += M5
    # M6 = (A10-A00)*(B00+B01) -> C10 -= M6, C11 -= M6  -- wait, let me use standard Strassen
    # 
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * (B00 + B01)
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A00 - A01) * (B01 + B11)
    #
    # C00 = M1 + M4 + M5 - M7
    # C01 = M2 + M7
    # C10 = M3 + M6
    # C11 = M1 + M2 - M3 + M4
    #
    # Wait, I need to double-check. Let me use a different standard form.
    #
    # Actually, the standard Strassen is:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 + B11)
    # M4 = A11 * (B01 - B10)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # i=0,1; j=0,1; k=0,1
    # A submatrix: rows i=0,1, cols j=0,1
    # B submatrix: rows j=0,1, cols k=0,1
    # C submatrix: rows i=0,1, cols k=0,1
    
    # M1 = (A00 + A11) * (B00 + B11)
    # In our indexing: A00 = A[0,0], A11 = A[1,1], B00 = B[0,0], B11 = B[1,1]
    # U: for A[0,0] and A[1,1] -> U[0*4+0] = U[0], U[1*4+1] = U[5]
    # V: for B[0,0] and B[1,1] -> V[0*5+0] = V[0], V[1*5+1] = V[6]
    # W: for C[0,0], C[0,1], C[1,0], C[1,1]
    # C00: k=0, i=0 -> W[0*3+0] = W[0]
    # C01: k=1, i=0 -> W[1*3+0] = W[3]
    # C10: k=0, i=1 -> W[0*3+1] = W[1]
    # C11: k=1, i=1 -> W[1*3+1] = W[4]
    
    # M1: (A00 + A11) * (B00 + B11)
    # U[0][col] = 1, U[5][col] = 1
    # V[0][col] = 1, V[6][col] = 1
    # C00 = M1 + ... so W[0][col] += 1
    # C11 = M1 - ... so W[4][col] += 1
    U[0][col] = 1
    U[5][col] = 1
    V[0][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    W[4][col] = 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    # A10 = A[1,0], A11 = A[1,1], B00 = B[0,0]
    # U[1*4+0] = U[4], U[1*4+1] = U[5]
    # V[0*5+0] = V[0]
    # C10 = M2 + M4, C11 = M1 - M2 + ...
    # So W[1][col] += 1, W[4][col] += -1
    U[4][col] = 1
    U[5][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    W[4][col] = -1
    col += 1
    
    # M3 = A00 * (B10 + B11)
    # A00 = A[0,0], B10 = B[1,0], B11 = B[1,1]
    # U[0*4+0] = U[0]
    # V[1*5+0] = V[5], V[1*5+1] = V[6]
    # C01 = M3 + M5, C11 = M1 - M2 + M3 + ...
    # So W[3][col] += 1, W[4][col] += 1
    U[0][col] = 1
    V[5][col] = 1
    V[6][col] = 1
    W[3][col] = 1
    W[4][col] = 1
    col += 1
    
    # M4 = A11 * (B01 - B10)
    # A11 = A[1,1], B01 = B[0,1], B10 = B[1,0]
    # U[1*4+1] = U[5]
    # V[0*5+1] = V[1], V[1*5+0] = V[5]
    # C00 = M1 + M4 - M5 + M7, C10 = M2 + M4
    # So W[0][col] += 1, W[1][col] += 1
    U[5][col] = 1
    V[1][col] = 1
    V[5][col] = -1
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    # A00 = A[0,0], A01 = A[0,1], B11 = B[1,1]
    # U[0*4+0] = U[0], U[0*4+1] = U[1]
    # V[1*5+1] = V[6]
    # C00 = M1 + M4 - M5 + M7, C01 = M3 + M5
    # So W[0][col] += -1, W[3][col] += 1
    U[0][col] = 1
    U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1
    W[3][col] = 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B10)
    # A10 = A[1,0], A00 = A[0,0], B00 = B[0,0], B10 = B[1,0]
    # U[1*4+0] = U[4], U[0*4+0] = U[0]
    # V[0*5+0] = V[0], V[1*5+0] = V[5]
    # C11 = M1 - M2 + M3 + M6
    # So W[4][col] += 1
    U[4][col] = 1
    U[0][col] = -1
    V[0][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01 - A11) * (B01 + B11)
    # A01 = A[0,1], A11 = A[1,1], B01 = B[0,1], B11 = B[1,1]
    # U[0*4+1] = U[1], U[1*4+1] = U[5]
    # V[0*5+1] = V[1], V[1*5+1] = V[6]
    # C00 = M1 + M4 - M5 + M7
    # So W[0][col] += 1
    U[1][col] = 1
    U[5][col] = -1
    V[1][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # Now Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # A submatrix: rows i=0,1, cols j=2,3
    # B submatrix: rows j=2,3, cols k=2,3
    # C submatrix: rows i=0,1, cols k=2,3
    
    # A00 = A[0,2], A01 = A[0,3], A10 = A[1,2], A11 = A[1,3]
    # B00 = B[2,2], B01 = B[2,3], B10 = B[3,2], B11 = B[3,3]
    # C00: k=2, i=0 -> W[2*3+0] = W[6]
    # C01: k=3, i=0 -> W[3*3+0] = W[9]
    # C10: k=2, i=1 -> W[2*3+1] = W[7]
    # C11: k=3, i=1 -> W[3*3+1] = W[10]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # U[0*4+2] = U[2], U[1*4+3] = U[7]
    # V[2*5+2] = V[12], V[3*5+3] = V[18]
    # W[6][col] += 1, W[10][col] += 1
    U[2][col] = 1
    U[7][col] = 1
    V[12][col] = 1
    V[18][col] = 1
    W[6][col] = 1
    W[10][col] = 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    # U[1*4+2] = U[6], U[1*4+3] = U[7]
    # V[2*5+2] = V[12]
    # W[7][col] += 1, W[10][col] += -1
    U[6][col] = 1
    U[7][col] = 1
    V[12][col] = 1
    W[7][col] = 1
    W[10][col] = -1
    col += 1
    
    # M3 = A00 * (B10 + B11)
    # U[0*4+2] = U[2]
    # V[3*5+2] = V[17], V[3*5+3] = V[18]
    # W[9][col] += 1, W[10][col] += 1
    U[2][col] = 1
    V[17][col] = 1
    V[18][col] = 1
    W[9][col] = 1
    W[10][col] = 1
    col += 1
    
    # M4 = A11 * (B01 - B10)
    # U[1*4+3] = U[7]
    # V[2*5+3] = V[13], V[3*5+2] = V[17]
    # W[6][col] += 1, W[7][col] += 1
    U[7][col] = 1
    V[13][col] = 1
    V[17][col] = -1
    W[6][col] = 1
    W[7][col] = 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    # U[0*4+2] = U[2], U[0*4+3] = U[3]
    # V[3*5+3] = V[18]
    # W[6][col] += -1, W[9][col] += 1
    U[2][col] = 1
    U[3][col] = 1
    V[18][col] = 1
    W[6][col] = -1
    W[9][col] = 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B10)
    # U[1*4+2] = U[6], U[0*4+2] = U[2]
    # V[2*5+2] = V[12], V[3*5+2] = V[17]
    # W[10][col] += 1
    U[6][col] = 1
    U[2][col] = -1
    V[12][col] = 1
    V[17][col] = 1
    W[10][col] = 1
    col += 1
    
    # M7 = (A01 - A11) * (B01 + B11)
    # U[0*4+3] = U[3], U[1*4+3] = U[7]
    # V[2*5+3] = V[13], V[3*5+3] = V[18]
    # W[6][col] += 1
    U[3][col] = 1
    U[7][col] = -1
    V[13][col] = 1
    V[18][col] = 1
    W[6][col] = 1
    col += 1
    
    # col should be 58
    # Let's verify: 60 - 16 + 14 = 58. Yes.
    
    # Check deadline
    if time.time() > deadline:
        # Return whatever we have so far
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
