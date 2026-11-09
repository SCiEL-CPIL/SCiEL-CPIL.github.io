import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, or numpy if allowed?
    # The prompt says "Python standard library, NumPy and SciPy are available".
    # The seed uses lists. I will use lists to be safe and match the seed's return type exactly.
    
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
                
                # Check deadline (unlikely to be hit, but required)
                if time.time() > deadline:
                    # Return whatever we have so far (which is complete for schoolbook)
                    # Since we are in the middle of filling, we might have incomplete data if we break early.
                    # But schoolbook is deterministic and fast.
                    pass
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
