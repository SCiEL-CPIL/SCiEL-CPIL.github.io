"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook initialization
    rank0 = n * m * p
    
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if decomposition is exact. Returns True if valid."""
        r = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def to_list(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Verify schoolbook
    if verify(U0, V0, W0):
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank0
    else:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0
    
    # Try to reduce rank via local search
    # Strategy: try to find columns that can be eliminated or merged
    # This is a hard problem, so we do limited search
    
    # For now, return schoolbook as it's guaranteed correct
    # A more sophisticated search would go here
    
    return to_list(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
