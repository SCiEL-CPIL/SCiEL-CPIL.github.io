"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_block_decomposition():
    """Try a block-based decomposition for 3x4x5."""
    n, m, p = DIMENSIONS
    # Try to use block structure
    # Split m=4 into 2+2, and use Strassen-like tricks
    # This is a heuristic attempt
    
    # For now, just return None to indicate no improvement found
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try various structured decompositions
    # 1. Try rank = max(n*m, m*p, p*n) = 20 with random search
    # 2. Try intermediate ranks
    
    # Random search for lower rank
    np.random.seed(42)
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # = 20
    
    for R in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        max_attempts = 50 if R < 40 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random vectors with small integer entries
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            W = np.random.randint(-2, 3, size=(p * n, R))
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = R
                break  # Found a valid decomposition for this rank, move to next
        
        # If we found a valid decomposition at this rank, we can stop
        # since we're going from low to high
        if best_rank == R:
            break
    
    return best_result
# EVOLVE-BLOCK-END
