import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1) > 1e-9:
                    return False, (i, j, k, val)
    # Check that non-matching positions are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                valid = (j1 == j2) and (i1 == i2) and (k1 == k2)
                val = np.sum(U[a] * V[b] * W[c])
                if valid:
                    if abs(val - 1) > 1e-9:
                        return False, (a, b, c, val)
                else:
                    if abs(val) > 1e-9:
                        return False, (a, b, c, val)
    return True, None

def _build_strassen_2x3x3():
    """
    Attempt to construct a rank-14 decomposition for M_{2,3,3} using a Strassen-style approach.
    
    We partition the 2x3 matrix A into two 2x2 blocks? No, that doesn't work directly.
    Instead, consider the multiplication of (2x3) * (3x3) = (2x3).
    
    Let's try to embed this into a 3x3 * 3x3 multiplication and use a known rank-19 or lower?
    Actually, the rank of M_{2,3,3} is conjectured to be 11, but we don't have a known construction.
    
    Let's try a different approach: Use the fact that M_{2,3,3} is a sub-tensor of M_{3,3,3}.
    Rank(M_{3,3,3}) is conjectured to be 19. If we can find a rank-19 decomposition of M_{3,3,3}
    that is "supported" on the 2x3x3 part, we might get a lower rank for M_{2,3,3}.
    
    However, a simpler heuristic: Try to combine Strassen's 7 multiplications for 2x2 blocks
    with the remaining 3x3 structure.
    
    Let's try a specific algebraic construction.
    A is 2x3, B is 3x3.
    Let A = [A1, A2] where A1, A2 are 2x2? No, A is 2x3.
    Let A = [[a11, a12, a13], [a21, a22, a23]]
    B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    This is complex. Let's try a random search with a structured initialization based on Strassen
    for the 2x2 sub-blocks, and see if we can find a rank-14 or 15 decomposition.
    
    Actually, let's just try a local search / random search with rank 14, 15, 16, 17.
    """
    n, m, p = 2, 3, 3
    # We will try to find a decomposition of rank < 18
    # This function is a placeholder for a more sophisticated search.
    # For now, we return None to indicate no known lower-rank decomposition.
    return None

def _try_random_search(n, m, p, target_rank, trials=5000, seed=42, time_limit=50):
    """Random search for a low-rank decomposition."""
    import random
    rng = random.Random(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        rank = target_rank
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        
        ok, _ = verify(U, V, W, n, m, p)
        if ok:
            return U, V, W, rank
    return None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 5  # safety margin
    
    # First, try known lower-rank constructions if available
    result = _build_strassen_2x3x3()
    if result is not None:
        U, V, W, rank = result
        ok, _ = verify(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Try random search with decreasing rank
    for target_rank in [14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 10:
            break
            
        result = _try_random_search(n, m, p, target_rank, time_limit=remaining_time - 5)
        if result is not None:
            U, V, W, rank = result
            ok, _ = verify(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
