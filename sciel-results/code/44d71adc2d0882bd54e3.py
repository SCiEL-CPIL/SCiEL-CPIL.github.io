import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """Return the lowest-rank exact decomposition found within the time budget."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    n_m = n * m      # 12
    m_p = m * p      # 24
    n_p = p * n      # 18
    schoolbook_rank = n * m * p  # 72

    # Build the schoolbook decomposition (guaranteed correct)
    U_sb = np.zeros((n_m, schoolbook_rank))
    V_sb = np.zeros((m_p, schoolbook_rank))
    W_sb = np.zeros((n_p, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    # Build the full tensor T as a flat array for verification
    T = np.zeros((n_m, m_p, n_p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_exact(U, V, W):
        """Check if U, V, W form an exact integer decomposition of T."""
        r = U.shape[1]
        T_rec = np.zeros((n_m, m_p, n_p), dtype=np.int64)
        for s in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, s].astype(np.int64), 
                               V[:, s].astype(np.int64), W[:, s].astype(np.int64))
        return np.array_equal(T_rec, T.astype(np.int64))

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try to find lower rank decompositions
    # Strategy: For each target rank, try random and structured initializations
    
    # First, try a structured approach: exploit the tensor structure
    # The tensor has 72 ones. Try to find linear dependencies among the rank-1 terms.
    
    # Approach: Try ranks from 72 down, with multiple random attempts
    # Use {-1, 0, 1} entries which are most likely for small-rank decompositions
    
    np.random.seed(42)
    
    for target_rank in range(schoolbook_rank - 1, 23, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple random initializations
        max_attempts = 50 if target_rank > 60 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Random integer matrices in {-1, 0, 1}
            U_try = np.random.randint(-1, 2, size=(n_m, target_rank))
            V_try = np.random.randint(-1, 2, size=(m_p, target_rank))
            W_try = np.random.randint(-1, 2, size=(n_p, target_rank))
            
            if verify_exact(U_try, V_try, W_try):
                best_rank = target_rank
                best_U = U_try.copy()
                best_V = V_try.copy()
                best_W = W_try.copy()
                found = True
                break
        
        if found:
            # Found a decomposition, try to go even lower
            continue
    
    # Convert to lists for return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()
    
    return {"U": U_ret, "V": V_ret, "W": W_ret}
