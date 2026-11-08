"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k), check if the product matches the expected value (1 if valid multiplication term)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry for matrix multiplication is 1 (since (A@B)[i,k] = sum_j A[i,j]*B[j,k])
                # So T[i,j,k] should be 1
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_strassen_like(n, m, p, max_rank=None):
    """
    Try to find a low-rank decomposition using structured patterns.
    For small dimensions, we can enumerate possible sparse patterns.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    # Start from schoolbook and try to reduce rank by finding linear dependencies
    # or by using algebraic tricks
    
    # For 2x4x7, known lower bounds suggest rank could be less than 56
    # Let's try a systematic search for rank < 56
    
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook as baseline
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    
    # Try to find improvements using a greedy approach
    # We'll attempt to find a decomposition with rank < 56
    # by trying various structured patterns
    
    # Approach: Use the fact that for matrix multiplication, we can sometimes
    # combine terms using algebraic identities
    
    # For 2x4x7, let's try rank values from 40 to 55
    for target_rank in range(40, best_rank):
        if target_rank >= best_rank:
            continue
        
        # Try a randomized search for a valid decomposition
        # This is heuristic - we try to find sparse patterns
        found = _search_rank(n, m, p, target_rank)
        if found:
            U, V, W = found
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = (U, V, W)
                break  # Found the lowest rank we're searching for
    
    return best_result

def _search_rank(n, m, p, target_rank, max_iters=1000):
    """
    Heuristic search for a rank-target_rank decomposition.
    Uses a greedy approach with random initialization.
    """
    np.random.seed(42)
    
    for _ in range(max_iters):
        # Initialize with random sparse patterns
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Assign random non-zero entries
        for c in range(target_rank):
            # Pick random positions
            ui = np.random.randint(0, n*m)
            vj = np.random.randint(0, m*p)
            wk = np.random.randint(0, p*n)
            
            # Ensure the indices are consistent (i from U matches i in W, j from U matches j in V, etc.)
            # U[i*m+j], V[j*p+k], W[k*n+i]
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            
            U[i*m+j, c] = 1
            V[j*p+k, c] = 1
            W[k*n+i, c] = 1
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # First, try to find a better decomposition
    result = _try_strassen_like(n, m, p)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
