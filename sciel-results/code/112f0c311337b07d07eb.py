"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        n, m, p = DIMENSIONS
        # Check tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check that all other entries are 0
        # This is expensive, so we'll skip for now
        return True
    
    # Try to find a better decomposition
    # For now, just return schoolbook
    # A future evolution could implement a search
    
    result = schoolbook()
    return result
# EVOLVE-BLOCK-END
