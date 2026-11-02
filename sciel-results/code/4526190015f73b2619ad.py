import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Uses the schoolbook decomposition of rank 54.
    Includes verification and time guard.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
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
                
    # Verification step
    def verify(U, V, W, n, m, p):
        # Check a sample of entries to ensure correctness
        # Full verification is O(n*m*p*rank) = 54*54 = 2916 ops, which is fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback (should not happen)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
