"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _target_tensor():
    """Return the 3x3x3 matrix multiplication tensor as a 3D array."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, T_target):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T_target[i,j,k]."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.int64)
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return np.array_equal(T, T_target)

def _rank26_construction():
    """Known valid rank-26 construction."""
    n, m, p = DIMENSIONS
    R = 26
    
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    # Strassen part (7 terms)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
    
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
    
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
    
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1
    
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1
    
    # Border terms (19 terms)
    border_indices = [
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    col = 7
    for i, j, k in border_indices:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
        
    assert col == R
    return U, V, W

def _search_lower_rank(r, time_limit, seed=0):
    """Search for a rank-r decomposition using structured random restarts."""
    rng = random.Random(seed)
    T_target = _target_tensor()
    n, m, p = DIMENSIONS
    nU, nV, nW = n * m, m * p, p * n
    
    start_time = time.time()
    max_attempts = 50000
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((nU, r), dtype=np.int64)
        V = np.zeros((nV, r), dtype=np.int64)
        W = np.zeros((nW, r), dtype=np.int64)
        
        for col in range(r):
            # Each rank-1 term: pick sparse support with values in {-1, 0, 1}
            # U: pick 1-3 nonzero entries
            nzU = rng.sample(range(nU), rng.randint(1, 3))
            for idx in nzU:
                U[idx, col] = rng.choice([-1, 1])
            
            nzV = rng.sample(range(nV), rng.randint(1, 3))
            for idx in nzV:
                V[idx, col] = rng.choice([-1, 1])
            
            nzW = rng.sample(range(nW), rng.randint(1, 3))
            for idx in nzW:
                W[idx, col] = rng.choice([-1, 1])
        
        if _verify_decomposition(U, V, W, T_target):
            return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try to find lower-rank decompositions
    for r in [23, 24, 25]:
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 10:
            break
        # Allocate time: more time for higher ranks (easier to find)
        time_alloc = min(remaining - 5, 40 if r == 25 else 25 if r == 24 else 15)
        if time_alloc < 2:
            break
        result = _search_lower_rank(r, time_alloc, seed=r * 7919)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to known rank-26 construction
    U, V, W = _rank26_construction()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
