"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(n, m, p, U, V, W):
    """Verify the decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    
    # Try a block-based decomposition
    # The key insight: we can partition the problem into blocks and use efficient
    # decompositions for each block, as long as the blocks don't overlap in a way
    # that causes interference.
    
    # For 3x4x5, let's try:
    # - Use Strassen (rank 7) for a 2x2x2 sub-problem
    # - Use schoolbook for the remaining parts
    
    # Partition:
    # Rows 0-1, Mids 0-1, Cols 0-1: 2x2x2 -> Strassen (rank 7)
    # All other (i,j,k) combinations: schoolbook (rank 1 each)
    
    # Total rank = 7 + (3*4*5 - 2*2*2) = 7 + (60 - 8) = 7 + 52 = 59
    # That's only 1 less than schoolbook (60). Not great.
    
    # Let's try a better partition:
    # Use Strassen for multiple 2x2x2 blocks that don't overlap in the middle dimension
    
    # Actually, the issue is that Strassen works on the full 2x2x2 tensor, not just
    # a subset of indices. We need to be more careful.
    
    # For now, return schoolbook as a safe baseline
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
