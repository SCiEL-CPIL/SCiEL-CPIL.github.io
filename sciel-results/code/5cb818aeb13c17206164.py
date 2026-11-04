import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _fast_verify(U, V, W, n=3, m=3, p=3):
    """Fast verification using tensor contraction.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    r = U_arr.shape[1]
    
    # Reshape: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
    U_reshaped = U_arr.reshape(n, m, r)
    V_reshaped = V_arr.reshape(m, p, r)
    W_reshaped = W_arr.reshape(p, n, r)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    return np.allclose(result, 1.0, atol=1e-9)


def _construct_schoolbook(n=3, m=3, p=3):
    """Construct the schoolbook rank-(n*m*p) decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _try_find_rank_r(r, n=3, m=3, p=3, time_budget=30.0, seed=42):
    """Try to find a rank-r decomposition using local search with {-1,0,1} entries."""
    start = time.time()
    np.random.seed(seed)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with structured pattern
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    
    # Start with schoolbook-like initialization for first min(r, n*m*p) columns
    nmp = n * m * p
    for c in range(min(r, nmp)):
        i = (c // (m * p)) % n
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    # For remaining columns, random initialization
    for c in range(min(r, nmp), r):
        U[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=n * m)
        V[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=m * p)
        W[:, c] = np.random.choice([-1.0, 0.0, 1.0], size=p * n)
    
    # Local search
    for iteration in range(50000):
        if time.time() - start > time_budget:
            break
        
        # Compute error
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        error = np.sum((result - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_error < 1e-9:
                return best_U, best_V, best_W
        
        # If we're getting close, try to polish
        if best_error < 1e-6:
            break
        
        # Random move: change one entry in one matrix
        choice = np.random.randint(3)
        if choice == 0:
            idx = np.random.randint(n * m)
            col = np.random.randint(r)
            old_val = U[idx, col]
            new_vals = [-1.0, 0.0, 1.0]
            new_val = np.random.choice([v for v in new_vals if v != old_val])
            U[idx, col] = new_val
        elif choice == 1:
            idx = np.random.randint(m * p)
            col = np.random.randint(r)
            old_val = V[idx, col]
            new_vals = [-1.0, 0.0, 1.0]
            new_val = np.random.choice([v for v in new_vals if v != old_val])
            V[idx, col] = new_val
        else:
            idx = np.random.randint(p * n)
            col = np.random.randint(r)
            old_val = W[idx, col]
            new_vals = [-1.0, 0.0, 1.0]
            new_val = np.random.choice([v for v in new_vals if v != old_val])
            W[idx, col] = new_val
    
    if best_error < 1e-9 and best_U is not None:
        return best_U, best_V, best_W
    
    return None


def _structured_rank23_search(n=3, m=3, p=3, time_budget=60.0):
    """
    Structured search for rank-23 decomposition.
    """
    start = time.time()
    r = 23
    
    # Initialize with schoolbook first 23 terms
    U, V, W = _construct_schoolbook(n, m, p)
    # Truncate to 23 columns
    U = U[:, :r].copy()
    V = V[:, :r].copy()
    W = W[:, :r].copy()
    
    best_error = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    np.random.seed(123)
    
    for iteration in range(100000):
        if time.time() - start > time_budget:
            break
        
        # Compute error
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        error = np.sum((result - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_error < 1e-9:
                return best_U, best_V, best_W
        
        if best_error < 1e-12:
            break
        
        # Try a random move
        choice = np.random.randint(3)
        if choice == 0:
            idx = np.random.randint(n * m)
            col = np.random.randint(r)
            old_val = U[idx, col]
            candidates = [v for v in [-1.0, 0.0, 1.0] if v != old_val]
            U[idx, col] = np.random.choice(candidates)
        elif choice == 1:
            idx = np.random.randint(m * p)
            col = np.random.randint(r)
            old_val = V[idx, col]
            candidates = [v for v in [-1.0, 0.0, 1.0] if v != old_val]
            V[idx, col] = np.random.choice(candidates)
        else:
            idx = np.random.randint(p * n)
            col = np.random.randint(r)
            old_val = W[idx, col]
            candidates = [v for v in [-1.0, 0.0, 1.0] if v != old_val]
            W[idx, col] = np.random.choice(candidates)
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    
    return None


def _polish_decomposition(U, V, W, n=3, m=3, p=3, max_iter=1000):
    """Polish a near-exact decomposition to make it exactly integer/half-integer."""
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    r = U.shape[1]
    
    # Round to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    if _fast_verify(U, V, W, n, m, p):
        return U, V, W
    
    # If rounding didn't work, try small perturbations
    for _ in range(max_iter):
        # Try flipping a sign or changing a value slightly
        choice = np.random.randint(3)
        if choice == 0:
            idx = np.random.randint(n * m)
            col = np.random.randint(r)
            U[idx, col] = -U[idx, col] if U[idx, col] != 0 else (0.5 if np.random.rand() > 0.5 else -0.5)
        elif choice == 1:
            idx = np.random.randint(m * p)
            col = np.random.randint(r)
            V[idx, col] = -V[idx, col] if V[idx, col] != 0 else (0.5 if np.random.rand() > 0.5 else -0.5)
        else:
            idx = np.random.randint(p * n)
            col = np.random.randint(r)
            W[idx, col] = -W[idx, col] if W[idx, col] != 0 else (0.5 if np.random.rand() > 0.5 else -0.5)
        
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        if _fast_verify(U, V, W, n, m, p):
            return U, V, W
    
    return None


def _try_rank20_search(n=3, m=3, p=3, time_budget=60.0):
    """
    Try to find a rank-20 decomposition using a more aggressive search.
    Uses {-1, -0.5, 0, 0.5, 1} as candidate values.
    """
    start = time.time()
    r = 20
    np.random.seed(42)
    
    # Initialize with schoolbook first 20 terms
    U, V, W = _construct_schoolbook(n, m, p)
    U = U[:, :r].copy()
    V = V[:, :r].copy()
    W = W[:, :r].copy()
    
    best_error = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Candidate values including half-integers
    candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for iteration in range(100000):
        if time.time() - start > time_budget:
            break
        
        # Compute error
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        error = np.sum((result - 1.0) ** 2)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_error < 1e-9:
                return best_U, best_V, best_W
        
        if best_error < 1e-12:
            break
        
        # Try a random move with half-integer candidates
        choice = np.random.randint(3)
        if choice == 0:
            idx = np.random.randint(n * m)
            col = np.random.randint(r)
            old_val = U[idx, col]
            valid_candidates = [v for v in candidates if v != old_val]
            U[idx, col] = np.random.choice(valid_candidates)
        elif choice == 1:
            idx = np.random.randint(m * p)
            col = np.random.randint(r)
            old_val = V[idx, col]
            valid_candidates = [v for v in candidates if v != old_val]
            V[idx, col] = np.random.choice(valid_candidates)
        else:
            idx = np.random.randint(p * n)
            col = np.random.randint(r)
            old_val = W[idx, col]
            valid_candidates = [v for v in candidates if v != old_val]
            W[idx, col] = np.random.choice(valid_candidates)
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    
    return None


def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed schoolbook rank-27 decomposition
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p  # 27
    
    # Try to find rank-20 decomposition (most aggressive)
    if time.time() - start_time < timeout - 20:
        time_budget = min(40.0, timeout - (time.time() - start_time) - 20)
        result = _try_rank20_search(n, m, p, time_budget)
        if result is not None:
            U, V, W = result
            if _fast_verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = 20
                return best_result
    
    # Try to find rank-23 decomposition
    if time.time() - start_time < timeout - 15:
        time_budget = min(35.0, timeout - (time.time() - start_time) - 15)
        result = _structured_rank23_search(n, m, p, time_budget)
        if result is not None:
            U, V, W = result
            if _fast_verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = 23
                return best_result
    
    # Try to find rank-24 or 25 decomposition as intermediate
    for target_rank in [24, 25, 26]:
        if time.time() - start_time < timeout - 5:
            time_budget = min(15.0, timeout - (time.time() - start_time) - 5)
            result = _try_find_rank_r(target_rank, n, m, p, time_budget)
            if result is not None:
                U, V, W = result
                if _fast_verify(U, V, W, n, m, p):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = target_rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
