"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy: Use block decomposition.
    Partition the 3x5 matrix A into blocks:
    - A11: 3x3 (rows 0-2, cols 0-2)
    - A12: 3x2 (rows 0-2, cols 3-4)
    Partition the 5x7 matrix B into blocks:
    - B11: 3x7 (rows 0-2, cols 0-6)
    - B21: 2x7 (rows 3-4, cols 0-6)
    
    C = A * B = [A11 A12] * [B11; B21] = A11*B11 + A12*B21
    
    We use:
    - Rank-23 Strassen-like decomposition for 3x3x3 (A11*B11_part where B11 is 3x3 part)
    - Rank-8 decomposition for 3x2x3 (A12*B21_part)
    - Rank-8 decomposition for 3x2x3 (A12*B21_part2)
    - Rank-23 decomposition for 3x3x3 (A11*B11_part2)
    
    Wait, let me think more carefully.
    
    A is 3x5, B is 5x7, C is 3x7.
    
    Let's partition columns of A and rows of B:
    A = [A_left (3x3) | A_right (3x2)]
    B = [B_top (3x7); B_bottom (2x7)]
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top: 3x3 * 3x7
    For A_right * B_bottom: 3x2 * 2x7
    
    Let's further partition columns of B_top and B_bottom:
    B_top = [B_top1 (3x3) | B_top2 (3x4)]
    B_bottom = [B_bottom1 (2x3) | B_bottom2 (2x4)]
    
    C = A_left * B_top1 + A_left * B_top2 + A_right * B_bottom1 + A_right * B_bottom2
    
    Block sizes:
    - A_left (3x3) * B_top1 (3x3) -> 3x3: rank 23
    - A_left (3x3) * B_top2 (3x4) -> 3x4: need decomposition
    - A_right (3x2) * B_bottom1 (2x3) -> 3x3: rank 8
    - A_right (3x2) * B_bottom2 (2x4) -> 3x4: need decomposition
    
    For 3x4 and 3x3 blocks, let's use schoolbook:
    - 3x3 * 3x4: rank = 3*3*4 = 36 (schoolbook)
    - 3x2 * 2x4: rank = 3*2*4 = 24 (schoolbook)
    
    Total with this partition: 23 + 36 + 8 + 24 = 91. Still better than 105.
    
    Let me try a different approach. Let's use the known rank-23 decomposition for 3x3x3.
    The Strassen algorithm for 3x3x3 has rank 23.
    
    Actually, let me look up the standard Strassen 3x3x3 decomposition.
    
    For 3x3 matrices, the tensor rank is 23. The decomposition uses 23 terms.
    
    Let me use a simpler approach: partition into 3x3x3 and 3x2x3 blocks.
    
    A (3x5) = [A1 (3x3) | A2 (3x2)]
    B (5x7) = [B1 (3x7); B2 (2x7)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (3x3 * 3x7): partition B1 into B1a (3x3) and B1b (3x4)
    A1*B1a: 3x3x3, rank 23
    A1*B1b: 3x3x4, rank 36 (schoolbook)
    
    For A2*B2 (3x2 * 2x7): partition B2 into B2a (2x3) and B2b (2x4)
    A2*B2a: 3x2x3, rank 8
    A2*B2b: 3x2x4, rank 24 (schoolbook)
    
    Total: 23 + 36 + 8 + 24 = 91
    
    Alternatively, we could use rank-8 for 3x2x3 and see if we can do better for the 3x3x4 case.
    
    Actually, let me just implement the 3x3x3 rank-23 decomposition and the 3x2x3 rank-8 decomposition,
    and use schoolbook for the remaining parts.
    
    The 3x2x3 rank-8 decomposition:
    For multiplying a 3x2 matrix by a 2x3 matrix to get a 3x3 matrix, the rank is 8.
    
    Let me define these decompositions explicitly.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build the decomposition block by block.
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (35, rank)
    # W: (p*n, rank) = (21, rank)
    
    # Block 1: A1 (3x3) * B1a (3x3) -> C1a (3x3)
    # This corresponds to rows 0-2 of A, cols 0-2 of A (which are rows 0-2 of B)
    # A1 is at A[0:3, 0:3], B1a is at B[0:3, 0:3], C1a is at C[0:3, 0:3]
    # In the tensor: U[i*m+j, c] for i in 0-2, j in 0-2
    # V[j*p+k, c] for j in 0-2, k in 0-2
    # W[k*n+i, c] for k in 0-2, i in 0-2
    
    # Block 2: A1 (3x3) * B1b (3x4) -> C1b (3x4)
    # B1b is at B[0:3, 3:7], C1b is at C[0:3, 3:7]
    # U[i*m+j, c] for i in 0-2, j in 0-2
    # V[j*p+k, c] for j in 0-2, k in 3-6
    # W[k*n+i, c] for k in 3-6, i in 0-2
    
    # Block 3: A2 (3x2) * B2a (2x3) -> C2a (3x3)
    # A2 is at A[0:3, 3:5], B2a is at B[3:5, 0:3], C2a is at C[0:3, 0:3]
    # U[i*m+j, c] for i in 0-2, j in 3-4
    # V[j*p+k, c] for j in 3-4, k in 0-2
    # W[k*n+i, c] for k in 0-2, i in 0-2
    
    # Block 4: A2 (3x2) * B2b (2x4) -> C2b (3x4)
    # B2b is at B[3:5, 3:7], C2b is at C[0:3, 3:7]
    # U[i*m+j, c] for i in 0-2, j in 3-4
    # V[j*p+k, c] for j in 3-4, k in 3-6
    # W[k*n+i, c] for k in 3-6, i in 0-2
    
    # Now I need the actual decompositions.
    # For 3x3x3 rank 23, I'll use a known decomposition.
    # For 3x2x3 rank 8, I'll use a known decomposition.
    
    # Let me use schoolbook for all blocks first to ensure correctness, then optimize.
    # Actually, let me just use schoolbook for everything to get a valid answer, then try to improve.
    
    # Schoolbook: rank = n*m*p = 3*5*7 = 105
    # But we can do better with block decomposition.
    
    # Let me implement the block decompositions with schoolbook for each block:
    # Block 1: 3x3x3 -> rank 27
    # Block 2: 3x3x4 -> rank 36
    # Block 3: 3x2x3 -> rank 18
    # Block 4: 3x2x4 -> rank 24
    # Total: 27 + 36 + 18 + 24 = 105 (same as full schoolbook)
    
    # To get a lower rank, I need to use non-schoolbook decompositions.
    # Let me use the rank-23 decomposition for 3x3x3 and rank-8 for 3x2x3.
    
    # For now, let me just return the schoolbook decomposition to ensure validity.
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
