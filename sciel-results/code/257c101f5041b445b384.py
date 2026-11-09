"""
Matrix multiplication 3,3,8 tensor rank decomposition.

Attempts to find a rank-minimal exact tensor decomposition for multiplying
a 3x3 matrix by a 3x8 matrix. Uses a search strategy to try to find rank < 72,
falling back to the schoolbook decomposition if no improvement is found.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s safety margin

def _check_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_search(n, m, p, max_rank_to_try, max_iterations=50000):
    """
    Try to find a decomposition with rank < schoolbook rank using local search.
    """
    schoolbook_rank = n * m * p
    best_result = None
    
    for target_rank in range(1, min(max_rank_to_try + 1, schoolbook_rank)):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Initialize with random values
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Random initialization
        for r in range(target_rank):
            U[:, r] = np.random.choice([-1, 0, 1], size=n*m)
            V[:, r] = np.random.choice([-1, 0, 1], size=m*p)
            W[:, r] = np.random.choice([-1, 0, 1], size=p*n)
        
        # Local search: try to fix violations
        for iteration in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Check if we have a valid decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists
                U_list = [[int(round(U[i, r])) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[int(round(V[i, r])) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[int(round(W[i, r])) for r in range(target_rank)] for i in range(p*n)]
                
                # Final verification
                if _check_decomposition(U_list, V_list, W_list, n, m, p, target_rank):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
            
            # Make a random perturbation
            if np.random.random() < 0.5:
                idx = np.random.randint(n*m)
                r = np.random.randint(target_rank)
                U[idx, r] = np.random.choice([-1, 0, 1])
            elif np.random.random() < 0.5:
                idx = np.random.randint(m*p)
                r = np.random.randint(target_rank)
                V[idx, r] = np.random.choice([-1, 0, 1])
            else:
                idx = np.random.randint(p*n)
                r = np.random.randint(target_rank)
                W[idx, r] = np.random.choice([-1, 0, 1])
    
    return best_result

def _structured_decomposition(n, m, p):
    """
    Try a structured approach: exploit the rectangular shape.
    For 3x3x8, try to use the 3x3x3 minimal rank decomposition extended to 8 columns.
    """
    # For now, this is a placeholder. The structured approach would use
    # known low-rank decompositions for smaller sub-problems.
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 3x3 matrix
    by a 3x8 matrix.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
              where each is a list of lists with integer entries.
    """
    n, m, p = DIMENSIONS
    
    # Try structured decomposition first
    result = _structured_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Try search for rank below schoolbook
    if time.time() - START_TIME < TIME_LIMIT - 10:
        result = _try_rank_search(n, m, p, max_rank_to_try=50, max_iterations=20000)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
