"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]
import time
import itertools
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Verify standard matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Target: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Einsum: 'ijc,jkc,kic->ijk'
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # Expected: 1.0 for all i, j, k
    expected = np.ones((n, m, p))
    
    if np.allclose(result, expected, atol=1e-9):
        return True
    return False

def _schoolbook(n, m, p):
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

def _try_random_search(n, m, p, rank, max_iters, seed):
    """Random search for a rank-rank decomposition with integer/half-integer entries."""
    rng = random.Random(seed)
    # Use small integer/half-integer values
    val_set = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    nm, mp, pn = n*m, m*p, p*n
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
        
        # Generate random sparse vectors
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for c in range(rank):
            # Each column: pick a few non-zero entries
            # Increase density slightly
            for _ in range(5):  
                ri = rng.randrange(nm)
                U[ri][c] = rng.choice(val_set)
            for _ in range(5):
                rj = rng.randrange(mp)
                V[rj][c] = rng.choice(val_set)
            for _ in range(5):
                rk = rng.randrange(pn)
                W[rk][c] = rng.choice(val_set)
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            # Convert to int or half-int representation
            def to_frac(x):
                x2 = round(x * 2)
                if x2 % 2 == 0:
                    return int(x2 // 2)
                else:
                    return x2 / 2
            U = [[to_frac(v) for v in row] for row in U]
            V = [[to_frac(v) for v in row] for row in V]
            W = [[to_frac(v) for v in row] for row in W]
            return {"U": U, "V": V, "W": W}
    return None

def _structured_search(n, m, p):
    """
    Try structured low-rank decompositions.
    For 3x3 * 3x6, try rank values below 54.
    """
    # The best known for 3x3 is 23. For 3x6, we can use two 3x3 blocks -> rank 46.
    # We try to find a rank 46 decomposition.
    # If we can't find it, we try 47, 48, etc.
    
    # We prioritize lower ranks.
    for rank in range(46, 54):
        if time.time() > _DEADLINE:
            break
        # Use more iterations for lower ranks
        max_iters = 5000 if rank < 50 else 2000
        result = _try_random_search(n, m, p, rank, max_iters, seed=rank*1000)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _DEADLINE
    
    # First, try structured/random search for lower rank
    result = _structured_search(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
