"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (naive) tensor decomposition for n×m × m×p multiplication.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    Convention: output index is W[k*n + i, r] where i is row of output, k is col of output.
    Actually the task says W[k*n+i, r] which means for each output element at position (i,k),
    the factor is in row k*n+i of W.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    # For each output element (i, k), the tensor coefficient for input (i, j, k) is 1.
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) valid, else 0.
    # Actually the multiplication tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k].
    # So for each (i, j, k), T[i, j, k] = 1.
    # The decomposition gives: for each r, U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Sum over r should equal 1 for all valid (i, j, k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_lasvegas_rank_search(n, m, p, max_rank, seed=42, time_limit=140.0):
    """Try to find a lower rank decomposition using randomized approaches.
    This is a heuristic search - may or may not find anything better than schoolbook.
    """
    start_time = time.time()
    import random
    random.seed(seed)
    
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # For 3x3x3, the Strassen algorithm gives rank 7 instead of 9.
    # Let's try to apply a similar idea to 3x3x7.
    # Actually, 3x3x7 can be seen as 7 blocks of 3x3x1, but that's not directly helpful.
    
    # Let's try a direct approach: for 3x3x3, we know rank 7 works.
    # For 3x3x7, the trivial upper bound is 3*3*7 = 63.
    # We could potentially do better, but we need explicit constructions.
    
    # Since we don't have explicit lower-rank decompositions for 3x3x7,
    # let's just return the schoolbook result.
    
    if time.time() - start_time > time_limit - 10:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = time.time() + 140.0  # 10 second safety margin
    
    # Start with schoolbook
    result = _schoolbook_decomposition(n, m, p)
    
    # Try to find a better decomposition
    if time.time() < deadline - 30:
        try:
            better = _try_lasvegas_rank_search(n, m, p, n*m*p, seed=12345, time_limit=min(60, deadline - time.time() - 5))
            if better is not None and len(better['U'][0]) < len(result['U'][0]):
                result = better
        except Exception:
            pass
    
    return result
# EVOLVE-BLOCK-END
