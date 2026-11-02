"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1  # Matrix mult tensor: T[i,j,k] = 1 if this is a valid product term
                # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
                # because C[i,k] = sum_j A[i,j]*B[j,k]
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_schoolbook():
    """Return the schoolbook decomposition as fallback."""
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

def _search_rank1_perturbations(n, m, p, target_rank, max_attempts=1000):
    """
    Search for a rank-target_rank decomposition by starting from schoolbook
    and trying to merge terms. This is a heuristic search.
    """
    import random
    random.seed(42)
    
    # Start with schoolbook
    base_rank = n * m * p
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    col_map = {}
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                col_map[(i, j, k)] = column
                column += 1
    
    # Try to find a lower rank decomposition
    # Strategy: Try random linear combinations to see if we can reduce rank
    # For now, return schoolbook as we need a more sophisticated search
    return None

def _structured_search(n, m, p):
    """
    Try structured decompositions that might achieve lower rank.
    For 3x4x8, try various factorization strategies.
    """
    # Try a simple structured approach:
    # Group columns of B and rows of A to create block decompositions
    
    # Strategy 1: Try rank = n*m*p - something by merging terms
    # The tensor T[i,j,k] = 1 for all i,j,k
    
    # Let's try a direct approach: enumerate small-rank candidates
    # For rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a hard constraint satisfaction problem.
    # Let's try a greedy approach: build up the decomposition term by term.
    
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But the actual lower bound for matrix mult is higher.
    
    # For now, let's try a few specific structured decompositions
    
    # Try: rank = 48 (half of schoolbook)
    # This might work if we can pair up terms
    
    # Actually, let's just try to verify if a slightly lower rank works
    # by attempting a simple search
    
    import itertools
    
    # Try rank values
    for target_rank in [48, 40, 36, 32]:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple heuristic: try random initializations and project
        success = _try_random_decomposition(n, m, p, target_rank, max_iters=100)
        if success is not None:
            return success
    
    return None

def _try_random_decomposition(n, m, p, target_rank, max_iters=100):
    """Try to find a decomposition using random initialization and projection."""
    import random
    random.seed(123)
    
    nm, mp, pn = n*m, m*p, p*n
    
    for attempt in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Initialize with random values from {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(pn)]
        
        # Check if this works
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Main entry point: find the lowest-rank decomposition."""
    n, m, p = DIMENSIONS
    
    # First, try structured search for lower rank
    result = _structured_search(n, m, p)
    if result is not None:
        # Verify before returning
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _try_schoolbook()
# EVOLVE-BLOCK-END
