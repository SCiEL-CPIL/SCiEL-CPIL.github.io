import time

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill

# EVOLVE-BLOCK-START
def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45
    
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

def construct():
    """
    Entry point with deadline guard. Returns the schoolbook rank-45 decomposition
    for 3x3 * 3x5 matrix multiplication.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > 150 - _DEADLINE_MARGIN:
        # Fallback: return zero matrices (invalid but avoids timeout)
        n, m, p = DIMENSIONS
        rank = n * m * p
        return {
            "U": [[0] * rank for _ in range(n * m)],
            "V": [[0] * rank for _ in range(m * p)],
            "W": [[0] * rank for _ in range(p * n)]
        }
    
    return _schoolbook_decomposition()
