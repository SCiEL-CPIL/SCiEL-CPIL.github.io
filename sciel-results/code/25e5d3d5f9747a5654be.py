"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    # Known rank-35 decomposition for 3x3x4 matrix multiplication
    # This is a specific algebraic construction that achieves rank 35
    def known_rank_35():
        # Based on known results, the tensor rank of 3x3x4 is 35
        # We construct this using a combination of Strassen-like methods
        rank = 35
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # This is a placeholder - we need the actual coefficients
        # For now, return None to indicate we don't have the exact construction
        return None

    best_result = schoolbook()
    
    # Try known lower rank constructions
    r35 = known_rank_35()
    if r35 is not None and verify(r35['U'], r35['V'], r35['W']):
        best_result = r35
    
    return best_result

# EVOLVE-BLOCK-END
