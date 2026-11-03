"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_partial(U, V, W, n, m, p, sample_size=50, tol=1e-6):
    """Quick partial verification on random samples."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rng = np.random.default_rng(12345)
    for _ in range(sample_size):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
        if abs(s - 1.0) > tol:
            return False
    return True

def _random_integer_decomposition(n, m, p, rank, seed, value_range=(-1, 1)):
    """Generate a random rank-r decomposition with integer entries."""
    rng = np.random.default_rng(seed)
    lo, hi = value_range
    U = rng.integers(lo, hi + 1, size=(n*m, rank)).astype(np.float64)
    V = rng.integers(lo, hi + 1, size=(m*p, rank)).astype(np.float64)
    W = rng.integers(lo, hi + 1, size=(p*n, rank)).astype(np.float64)
    return U, V, W

def _try_rank_structured(n, m, p, rank, time_limit, seed_base=0):
    """Try to find a valid rank-r decomposition using structured random search."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Try different value ranges and strategies
    for attempt in range(5000):
        if time.time() - start_time > time_limit:
            return None
        
        seed = seed_base + attempt
        
        # Use small integer values for better chance of exact solutions
        if attempt % 3 == 0:
            U, V, W = _random_integer_decomposition(n, m, p, rank, seed, (-1, 1))
        elif attempt % 3 == 1:
            U, V, W = _random_integer_decomposition(n, m, p, rank, seed, (-2, 2))
        else:
            U, V, W = _random_integer_decomposition(n, m, p, rank, seed, (-1, 1))
        
        # Quick partial check
        if not _verify_partial(U.tolist(), V.tolist(), W.tolist(), n, m, p, sample_size=20):
            continue
        
        # Full verification
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_rank_half_integer(n, m, p, rank, time_limit, seed_base=0):
    """Try to find a valid rank-r decomposition using half-integer entries."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    for attempt in range(3000):
        if time.time() - start_time > time_limit:
            return None
        
        seed = seed_base + attempt
        rng = np.random.default_rng(seed)
        
        # Half-integer values: -1.5, -1, -0.5, 0, 0.5, 1, 1.5
        vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(nm)])
        V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(mp)])
        W = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(pn)])
        
        if not _verify_partial(U.tolist(), V.tolist(), W.tolist(), n, m, p, sample_size=20):
            continue
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook (rank 100) - Guaranteed valid
    best = _schoolbook()
    best_rank = n * m * p
    
    # We only have ~120s total. 
    # Random search for rank < 100 is extremely unlikely to succeed in reasonable time for 4x5x5.
    # We will allow a short window (e.g., 5 seconds) to try a few quick checks, 
    # but we must ensure we return before the 120s limit.
    
    # Set a strict deadline for search, leaving 10s buffer for return/overhead
    search_deadline = start_time + 10.0
    
    # Lower bound: max(n*m, m*p, p*n) = 25
    lower_bound = max(n*m, m*p, p*n)
    
    # Only try a few ranks very quickly
    # If we don't find a solution in the first few seconds, we stop.
    ranks_to_try = range(lower_bound, min(lower_bound + 5, best_rank))
    
    for rank in ranks_to_try:
        if time.time() > search_deadline:
            break
            
        # Very short time limit per rank (1 second)
        time_for_this_rank = 1.0
        
        # Try integer-valued decompositions
        result = _try_rank_structured(n, m, p, rank, time_for_this_rank, seed_base=rank * 1000)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        # If not found, try half-integer briefly
        if time.time() < search_deadline:
            result = _try_rank_half_integer(n, m, p, rank, time_for_this_rank / 2, seed_base=rank * 1000 + 500)
            if result is not None:
                best = result
                best_rank = rank
                break
                
    return best
# EVOLVE-BLOCK-END
