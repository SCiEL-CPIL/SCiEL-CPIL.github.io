"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _target_tensor():
    """Return the 3x3x3 matrix multiplication tensor as a 3D array."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.int64)
    return T

def _verify_decomposition_fast(U, V, W, T_target):
    """
    Verify decomposition using vectorized einsum.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Reshape to (n, m, R), (m, p, R), (p, n, R)
    Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    """
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Reshape to 3D tensors for einsum
    # U is indexed by i, j -> shape (n, m, R)
    # V is indexed by j, k -> shape (m, p, R)
    # W is indexed by k, i -> shape (p, n, R)
    
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # T_recon[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    return np.array_equal(T_recon, T_target)

def _rank26_construction():
    """Known valid rank-26 construction (Strassen 2x2 core + 19 border terms)."""
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
    
    # Strassen part (7 terms) for indices i,j,k in {0,1}
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
    
    # m3 = a00(b01 - b11)
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    # m4 = a11(b10 - b00)
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
    
    # m5 = (a00 + a01)b11
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1
    
    # Border terms (19 terms)
    # These cover all (i,j,k) not in the 2x2x2 core
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
    """
    Search for a rank-r decomposition using vectorized verification.
    """
    rng = random.Random(seed)
    T_target = _target_tensor()
    n, m, p = DIMENSIONS
    nU, nV, nW = n * m, m * p, p * n
    
    start_time = time.time()
    max_attempts = 200000 # Increased due to fast verification
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((nU, r), dtype=np.int64)
        V = np.zeros((nV, r), dtype=np.int64)
        W = np.zeros((nW, r), dtype=np.int64)
        
        for col in range(r):
            # Random sparse support
            # Heuristic: Strassen terms have ~2 non-zeros per matrix.
            # Border terms have 1.
            # We sample sparsity 1-3.
            
            nzU = rng.sample(range(nU), rng.randint(1, 3))
            for idx in nzU:
                U[idx, col] = rng.choice([-1, 1])
            
            nzV = rng.sample(range(nV), rng.randint(1, 3))
            for idx in nzV:
                V[idx, col] = rng.choice([-1, 1])
            
            nzW = rng.sample(range(nW), rng.randint(1, 3))
            for idx in nzW:
                W[idx, col] = rng.choice([-1, 1])
        
        if _verify_decomposition_fast(U, V, W, T_target):
            return U, V, W
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Returns dict with U, V, W.
    """
    start_time = time.time()
    deadline = start_time + 135.0 # 15s safety margin
    
    # 1. Attempt to find a lower rank solution (23, 24, 25)
    # We prioritize 23 as it is the theoretical minimum, but 24/25 are easier.
    # Given the difficulty, we allocate short time to each.
    
    # Try Rank 23 (very hard)
    remaining = deadline - time.time()
    if remaining > 10:
        time_alloc = min(10, remaining - 5)
        result = _search_lower_rank(23, time_alloc, seed=23)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Rank 24
    remaining = deadline - time.time()
    if remaining > 10:
        time_alloc = min(15, remaining - 5)
        result = _search_lower_rank(24, time_alloc, seed=24)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Rank 25
    remaining = deadline - time.time()
    if remaining > 10:
        time_alloc = min(20, remaining - 5)
        result = _search_lower_rank(25, time_alloc, seed=25)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Fall back to the known Rank-26 construction
    U, V, W = _rank26_construction()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
