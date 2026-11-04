"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j][c] = 1
                best_V[j * p + k][c] = 1
                best_W[k * n + i][c] = 1
    
    # Try to reduce rank by merging terms with linear dependencies
    # For each pair of columns, check if we can merge them
    # A simpler approach: try smaller ranks with structured patterns
    
    # Try rank = n*m*p - reduction
    # Known: for M_{n,m,p}, rank <= n*m*p - m*min(n,p) + something
    # Let's try a few candidate ranks and verify
    
    for target_rank in range(best_rank - 1, max(50, best_rank - 30), -1):
        if time.time() > deadline:
            break
        
        # Try a structured decomposition: use rank-r factorization of middle
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Split j into two groups and use an intermediate
        # For m=4, try r=3: C[i,k] = sum_{j=1}^{3} A[i,j'] B[j',k] where j' is a linear combo
        # This doesn't directly work for exact decomposition
        
        # Instead, try to find a valid decomposition of target_rank
        # by checking if we can set up a system
        # This is complex; for now, just verify schoolbook and return
        pass
    
    # Return best found (schoolbook for now)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
