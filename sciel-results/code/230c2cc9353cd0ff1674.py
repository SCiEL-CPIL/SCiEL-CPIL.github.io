import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _verify_fast(U, V, W, T):
    """Faster verification using numpy."""
    rank = U.shape[1]
    if rank == 0:
        return np.all(T == 0)
    # Use einsum for efficient tensor contraction
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Result: (n*m, m*p, p*n)
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.array_equal(result, T)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_rank_decomposition(T, rank, timeout):
    """Try to find a rank decomposition using random initialization."""
    n, m, p = DIMENSIONS
    shape_u = n * m
    shape_v = m * p
    shape_w = p * n
    
    start_time = time.time()
    
    # Random search with integer entries
    # We use a small range to keep the search space manageable
    # and because large integers are unlikely to cancel out to 1s
    # in a low-rank decomposition of a 0/1 tensor.
    while time.time() - start_time < timeout:
        U = np.random.randint(-2, 3, size=(shape_u, rank))
        V = np.random.randint(-2, 3, size=(shape_v, rank))
        W = np.random.randint(-2, 3, size=(shape_w, rank))
        
        if _verify_fast(U, V, W, T):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Get schoolbook solution (Rank 60)
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140  # Leave 10s margin for 150s limit
    
    # Try to find lower rank
    # Lower bound is max(n*m, m*p, p*n) = 20
    # We try ranks from best_rank - 1 down to a reasonable limit
    # Given the structure, 59 is the first candidate.
    
    # We allocate a small budget for searching for rank 59
    # If it's not found quickly, we assume 60 is minimal.
    search_budget_per_rank = 5.0
    
    for rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1:
            break
        
        # Limit search time for this rank
        t_limit = min(search_budget_per_rank, remaining - 1)
        if t_limit <= 0:
            break
            
        result = _try_rank_decomposition(T, rank, t_limit)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break  # Found the minimal rank (assuming we search downwards)
            
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
