import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    if U_np.shape[1] != V_np.shape[1] or U_np.shape[1] != W_np.shape[1]:
        return False
        
    rank = U_np.shape[1]
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
                
    # Check zero entries
    # Precompute the set of valid (u, v, w)
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid.add((u, v, w))
                
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in valid:
                    val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
    return True

def _schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U.tolist(), V.tolist(), W.tolist()

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    """
    Try to find a decomposition of target_rank.
    Strategy:
    1. Random initialization with integer/half-integer entries.
    2. Verification.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # We will try to find a solution with rank < nm*mp*p
    # Since the schoolbook rank is nm*p (which is 36), we try 35 down to 30.
    
    # Use a deterministic seed for reproducibility in the search phase
    # But the task doesn't require reproducibility, just speed.
    # We use random state for variety.
    
    for attempt in range(100):
        if time.time() > time_limit:
            break
            
        # Initialize with small random integers/half-integers
        # This is a shot in the dark, but maybe we hit the solution.
        # We bias towards sparse matrices as they are more likely to satisfy exact constraints?
        # Actually, random dense is fine.
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a better decomposition
    # Start from rank 35 down to 30
    # The schoolbook rank is 36.
    for r in range(35, 29, -1):
        if time.time() > deadline - 10:
            break
        result = _try_find_decomposition(n, m, p, r, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
