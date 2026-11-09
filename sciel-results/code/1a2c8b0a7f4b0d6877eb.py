"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Strategy: Split A (3x4) into A1 (3x3) and A2 (3x1).
    # Split B (4x6) into B1 (3x6) and B2 (1x6).
    # C = A1*B1 + A2*B2.
    # A1*B1 (3x3x6) is split into A1*B1a (3x3x3) and A1*B1b (3x3x3).
    # Use schoolbook rank 27 for each 3x3x3 block.
    # A2*B2 (3x1x6) has rank 1.
    # Total rank = 27 + 27 + 1 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 12 x 55
    V = [[0] * rank for _ in range(m * p)]  # 24 x 55
    W = [[0] * rank for _ in range(p * n)]  # 18 x 55
    
    col = 0
    
    # Helper to add a schoolbook 3x3x3 decomposition
    # i_offset: offset for i in A (0..2)
    # j_offset: offset for j in A/B (0..2 for A1, 3 for A2)
    # k_offset: offset for k in B/C (0..2 for B1a, 3..5 for B1b)
    # Note: A is 3x4, so j is 0..3. B is 4x6, so j is 0..3, k is 0..5.
    # W index is k*n + i.
    
    def add_3x3x3(i_base, j_base, k_base):
        nonlocal col
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = (i_base + i) * m + (j_base + j)
                    v_idx = (j_base + j) * p + (k_base + k)
                    w_idx = (k_base + k) * n + (i_base + i)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
                    if col >= rank:
                        raise RuntimeError("Rank overflow")

    # Block 1: A1 (i 0-2, j 0-2) * B1a (j 0-2, k 0-2)
    add_3x3x3(0, 0, 0)
    
    # Block 2: A1 (i 0-2, j 0-2) * B1b (j 0-2, k 3-5)
    add_3x3x3(0, 0, 3)
    
    # Block 3: A2 (i 0-2, j 3) * B2 (j 3, k 0-5)
    # Rank 1 decomposition for 3x1x6
    # T[i, 3, k] = 1
    for i in range(3):
        for k in range(6):
            u_idx = i * m + 3
            v_idx = 3 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
    col += 1
    
    if col != rank:
        # If col < rank, we have unused columns, which is fine, but we should trim or keep zeros.
        # The interface expects shape (n*m, rank). Zeros are valid.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
