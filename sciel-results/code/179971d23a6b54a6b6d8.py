"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_random_decomposition(n, m, p, target_rank, rng, time_budget=5.0):
    """
    Try to find a valid decomposition at target_rank using random search
    with values in {-1, 0, 1} and {-0.5, 0, 0.5, 1, 1.5, 2}.
    """
    start_time = time.time()
    values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for attempt in range(200):
        if time.time() - start_time > time_budget:
            break
        
        U_t = np.zeros((n * m, target_rank), dtype=float)
        V_t = np.zeros((m * p, target_rank), dtype=float)
        W_t = np.zeros((p * n, target_rank), dtype=float)
        
        for r in range(target_rank):
            # Each column is a random vector
            for idx in range(n * m):
                if rng.random() < 0.6:
                    U_t[idx, r] = rng.choice(values)
            for idx in range(m * p):
                if rng.random() < 0.6:
                    V_t[idx, r] = rng.choice(values)
            for idx in range(p * n):
                if rng.random() < 0.6:
                    W_t[idx, r] = rng.choice(values)
        
        if _verify_decomposition(U_t, V_t, W_t, n, m, p):
            return U_t, V_t, W_t, True
    
    return None, None, None, False

def _structured_search(n, m, p, target_rank, rng, time_budget=10.0):
    """
    Structured search: try to build decomposition term by term,
    ensuring each term contributes to covering the required (i,j,k) triples.
    """
    start_time = time.time()
    
    # Track which (i,j,k) triples are covered
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
    
    U_t = np.zeros((n * m, target_rank), dtype=float)
    V_t = np.zeros((m * p, target_rank), dtype=float)
    W_t = np.zeros((p * n, target_rank), dtype=float)
    
    # Try different strategies for assigning values
    for strategy in range(5):
        if time.time() - start_time > time_budget:
            break
        
        U_t[:] = 0
        V_t[:] = 0
        W_t[:] = 0
        
        for r in range(target_rank):
            if time.time() - start_time > time_budget:
                return None, None, None, False
            
            # Choose a pattern for this term
            # Strategy 0: sparse random
            if strategy == 0:
                for idx in range(n * m):
                    if rng.random() < 0.3:
                        U_t[idx, r] = rng.choice([-1, 1])
                for idx in range(m * p):
                    if rng.random() < 0.3:
                        V_t[idx, r] = rng.choice([-1, 1])
                for idx in range(p * n):
                    if rng.random() < 0.3:
                        W_t[idx, r] = rng.choice([-1, 1])
            
            # Strategy 1: block-based
            elif strategy == 1:
                i_block = r % n
                j_block = (r // n) % m
                k_block = (r // (n*m)) % p
                U_t[i_block*m + j_block, r] = 1
                V_t[j_block*p + k_block, r] = 1
                W_t[k_block*n + i_block, r] = 1
            
            # Strategy 2: diagonal-like
            elif strategy == 2:
                idx = r % (n*m)
                i, j = divmod(idx, m)
                k = r % p
                U_t[i*m + j, r] = 1
                V_t[j*p + k, r] = 1
                W_t[k*n + i, r] = 1
            
            # Strategy 3: mixed
            elif strategy == 3:
                for idx in range(n * m):
                    if rng.random() < 0.4:
                        U_t[idx, r] = rng.choice([-1, 0, 1])
                for idx in range(m * p):
                    if rng.random() < 0.4:
                        V_t[idx, r] = rng.choice([-1, 0, 1])
                for idx in range(p * n):
                    if rng.random() < 0.4:
                        W_t[idx, r] = rng.choice([-1, 0, 1])
            
            # Strategy 4: half-integers
            elif strategy == 4:
                for idx in range(n * m):
                    if rng.random() < 0.5:
                        U_t[idx, r] = rng.choice([-0.5, 0.5, 1.0])
                for idx in range(m * p):
                    if rng.random() < 0.5:
                        V_t[idx, r] = rng.choice([-0.5, 0.5, 1.0])
                for idx in range(p * n):
                    if rng.random() < 0.5:
                        W_t[idx, r] = rng.choice([-0.5, 0.5, 1.0])
        
        if _verify_decomposition(U_t, V_t, W_t, n, m, p):
            return U_t, V_t, W_t, True
    
    return None, None, None, False

def _local_search_from_schoolbook(n, m, p, U_sb, V_sb, W_sb, target_rank, rng, time_budget=15.0):
    """
    Start from schoolbook and try to merge terms to reduce rank.
    """
    start_time = time.time()
    
    # Try to find a subset of schoolbook terms that can be replaced
    # by fewer terms with non-trivial values
    
    U_t = np.zeros((n * m, target_rank), dtype=float)
    V_t = np.zeros((m * p, target_rank), dtype=float)
    W_t = np.zeros((p * n, target_rank), dtype=float)
    
    # Initialize with first target_rank schoolbook terms
    for r in range(min(target_rank, n*m*p)):
        i, j, k = divmod(r, m*p)
        k, j = divmod(j, p)
        i = r // (m*p)
        j = (r // p) % m
        k = r % p
        U_t[i*m + j, r] = 1
        V_t[j*p + k, r] = 1
        W_t[k*n + i, r] = 1
    
    # Try to modify terms to cover more triples
    for iteration in range(100):
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random term to modify
        r = rng.randint(0, target_rank - 1)
        
        # Try different values for this term
        for _ in range(10):
            U_t[:, r] = rng.choice([-1, 0, 1], size=n*m)
            V_t[:, r] = rng.choice([-1, 0, 1], size=m*p)
            W_t[:, r] = rng.choice([-1, 0, 1], size=p*n)
            
            if _verify_decomposition(U_t, V_t, W_t, n, m, p):
                return U_t, V_t, W_t, True
    
    return None, None, None, False

def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook solution as a guaranteed fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = 16
    min_rank = max(n * m, m * p, p * n)
    
    # Try ranks from just below schoolbook down to min_rank
    for target_rank in range(rank_sb - 1, min_rank - 1, -1):
        if time.time() - start_time > 110:  # Safety margin
            break
        
        remaining = 110 - (time.time() - start_time)
        if remaining < 3:
            break
        
        rng = random.Random(target_rank * 1000 + 42)
        
        # Try structured search
        U_t, V_t, W_t, found = _structured_search(n, m, p, target_rank, rng, 
                                                    time_budget=min(8, remaining * 0.4))
        if found:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
        
        if time.time() - start_time > 110:
            break
        
        # Try random search
        remaining = 110 - (time.time() - start_time)
        if remaining < 3:
            break
        U_t, V_t, W_t, found = _try_random_decomposition(n, m, p, target_rank, rng,
                                                          time_budget=min(5, remaining * 0.4))
        if found:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
        
        if time.time() - start_time > 110:
            break
        
        # Try local search from schoolbook
        remaining = 110 - (time.time() - start_time)
        if remaining < 3:
            break
        U_t, V_t, W_t, found = _local_search_from_schoolbook(n, m, p, U_sb, V_sb, W_sb,
                                                               target_rank, rng,
                                                               time_budget=min(8, remaining * 0.5))
        if found:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
    
    # Verify the best solution
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Convert to list format
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
