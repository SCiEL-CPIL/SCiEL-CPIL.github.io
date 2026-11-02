import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook():
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

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Efficient check: for each column c, the contribution is outer product of vectors
    # u_c (length n*m), v_c (length m*p), w_c (length p*n)
    # T += u_c ⊗ v_c ⊗ w_c reshaped appropriately
    
    # Simpler: just compute all 96 entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using 8 independent 2x2x2 Strassen blocks. Rank 56."""
    U2, V2, W2 = _strassen_2x2x2()
    
    U4 = [[0]*56 for _ in range(16)]
    V4 = [[0]*56 for _ in range(16)]
    W4 = [[0]*56 for _ in range(16)]
    
    def add_block(u_map, v_map, w_map, col_offset):
        for i in range(4):
            for c in range(7):
                U4[u_map[i]][col_offset + c] += U2[i][c]
                V4[v_map[i]][col_offset + c] += V2[i][c]
                W4[w_map[i]][col_offset + c] += W2[i][c]
    
    # 8 blocks for 4x4x4
    # Block (row_block, col_block) for A and B, output to C block
    blocks = [
        # (u_map, v_map, w_map, offset)
        ([0,1,4,5], [0,1,4,5], [0,1,4,5], 0),      # A11*B11 -> C11
        ([2,3,6,7], [8,9,12,13], [0,1,4,5], 7),    # A12*B21 -> C11
        ([0,1,4,5], [2,3,6,7], [2,3,6,7], 14),     # A11*B12 -> C12
        ([2,3,6,7], [10,11,14,15], [2,3,6,7], 21), # A12*B22 -> C12
        ([8,9,12,13], [0,1,4,5], [8,9,12,13], 28), # A21*B11 -> C21
        ([10,11,14,15], [8,9,12,13], [8,9,12,13], 35), # A22*B21 -> C21
        ([8,9,12,13], [2,3,6,7], [10,11,14,15], 42), # A21*B12 -> C22
        ([10,11,14,15], [10,11,14,15], [10,11,14,15], 49) # A22*B22 -> C22
    ]
    
    for u_map, v_map, w_map, offset in blocks:
        add_block(u_map, v_map, w_map, offset)
    
    return U4, V4, W4

def _build_3x4x8_strassen():
    """
    Build 3x4x8 decomposition using Strassen-like approach.
    
    Strategy: Pad 3x4 to 4x4, split 4x8 into two 4x4 blocks.
    Use 4x4x4 Strassen decomposition twice.
    Rank = 2 * 56 = 112.
    
    Wait, that's worse than schoolbook (96). Let me think differently.
    
    Alternative: Use 2x2x2 Strassen more directly.
    Split 3x4 into blocks and 4x8 into blocks.
    
    Actually, let's try a different approach:
    3x4 * 4x8:
    Split A (3x4) into:
      - Top 2 rows: 2x4
      - Bottom 1 row: 1x4
    
    Split B (4x8) into:
      - Left 4 cols: 4x4
      - Right 4 cols: 4x4
    
    This is getting complicated. Let's just use the padding approach
    but verify if it's actually better.
    
    Actually, 112 > 96, so this is worse. Let's not use this.
    
    For now, return schoolbook. The real improvement will come from
    the greedy rank-reduction search below.
    """
    return _schoolbook()

def _greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    
    Each column c defines a rank-1 tensor: u_c ⊗ v_c ⊗ w_c.
    If we can find a linear combination of columns that still preserves
    the tensor decomposition, we can eliminate a column.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 96).
    2. For each pair of columns, check if one can be expressed as a linear
       combination of others (in the appropriate sense).
    3. This is complex, so we use a simpler heuristic:
       - Randomly perturb the decomposition
       - Try to solve for a lower-rank equivalent
    
    Actually, a better approach: use the fact that for matrix multiplication,
    there are known algebraic identities (Strassen-like) that reduce rank.
    
    Let's implement a search over known Strassen-type constructions.
    """
    best = {"U": U, "V": V, "W": W}
    best_rank = len(U[0])
    
    # Try the 4x4x4 Strassen construction (rank 56 for 4x4x4)
    # But we need to adapt it to 3x4x8.
    
    # Alternative: directly search for a good decomposition
    # using random initialization + local optimization
    
    # For now, just return the input if no improvement found
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook()
    
    # Try to find a better decomposition
    # The key insight: for 3x4x8, we can use Strassen-like methods
    # to reduce the rank below 96.
    
    # Let's try the direct Strassen construction for rectangular matrices.
    # 
    # For 3x4 * 4x8:
    # We can pad to 4x4 * 4x8, then split 4x8 into two 4x4.
    # Use 4x4x4 Strassen (rank 56) twice: total rank 112. Worse.
    #
    # Better: Use 2x2x2 Strassen more cleverly.
    #
    # Actually, the best known approach for rectangular matrix multiplication
    # is to use the square case results with padding.
    #
    # For 3x4x8, one approach:
    # - Pad A to 4x4 (add zero row)
    # - Use 4x4x4 decomposition with rank 49 (if we can find it)
    # - But 4x4x4 best known is around 49? Let me check.
    #   Actually, 4x4x4 can be done with rank 49 using Strassen recursion:
    #   T(4) = 7 * T(2) = 7 * 7 = 49? No, that's not right either.
    #
    # The correct recursion: To multiply two 4x4 matrices, split into 2x2 blocks.
    # You need 8 block multiplications of 2x2 matrices (schoolbook).
    # With Strassen, you can do it in 7 multiplications of 2x2 matrices? No.
    # Strassen reduces 2x2x2 to 7, but for 4x4x4, you still need to handle
    # the 8 block products. Strassen's algorithm for 4x4 would be:
    # T(4) = 7 * T(2) if we could reduce 8 to 7, but we can't directly.
    #
    # Actually, Strassen's algorithm works recursively:
    # T(n) = 7 * T(n/2) for n even.
    # T(2) = 7
    # T(4) = 7 * 7 = 49
    # T(8) = 7 * 49 = 343
    #
    # So rank(4x4x4) <= 49!
    #
    # For 3x4x8:
    # Pad A to 4x4, B to 4x8 (no padding needed for B since it's already 4x8).
    # But we need 4x4 * 4x8.
    # Split 4x8 into two 4x4: B1, B2.
    # Compute C1 = A' * B1 using 4x4x4 rank 49.
    # Compute C2 = A' * B2 using 4x4x4 rank 49.
    # Total rank: 98. Still worse than 96.
    #
    # Hmm. What if we use a different splitting?
    # 4x4 * 4x8:
    # Split A into 2x4 and 2x4 (two 2x4 blocks).
    # Split B into 4x4 and 4x4.
    # C1 = A_top * B1 (2x4 * 4x4 = 2x4)
    # C2 = A_top * B2 (2x4 * 4x4 = 2x4)
    # C3 = A_bot * B1 (2x4 * 4x4 = 2x4)
    # C4 = A_bot * B2 (2x4 * 4x4 = 2x4)
    #
    # Each 2x4 * 4x4 multiplication:
    # Pad to 4x4 * 4x4 (rank 49) -> too expensive.
    # Or use 2x4 * 4x4 directly.
    #
    # 2x4 * 4x4:
    # A is 2x4, B is 4x4, C is 2x4.
    # Schoolbook: 2*4*4 = 32.
    # Can we do better?
    # Pad A to 4x4, use 4x4x4 rank 49. But we only need 2 rows of C.
    # The rank is still 49 for the full 4x4x4, but we only use part of it.
    # However, the rank of the decomposition is the number of scalar multiplications,
    # which is 49 regardless of whether we use all output entries.
    #
    # So 4 * 49 = 196. Worse.
    #
    # Let's try yet another approach:
    # 3x4 * 4x8.
    # Use 2x2x2 Strassen (rank 7) as a building block.
    #
    # Split A (3x4) into:
    #   A1: rows 0-1, cols 0-1 (2x2)
    #   A2: rows 0-1, cols 2-3 (2x2)
    #   A3: row 2, cols 0-3 (1x4) -> pad to 2x2 by adding zero row
    #
    # This is getting too complex. Let's just return schoolbook for now.
    # The greedy search might find something, but it's unlikely to beat 96
    # without a known construction.
    
    # Try a simple random search for small improvements
    random.seed(42)
    
    # We'll try to find a rank < 96 decomposition by random perturbation
    # and checking if it's valid. This is very unlikely to work, but let's try.
    
    # Actually, let's just return schoolbook. It's safe and correct.
    return result
# EVOLVE-BLOCK-END
