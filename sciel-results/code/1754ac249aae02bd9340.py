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
    # T is (n_m, m_p, n_p)
    T = np.zeros((n_m, m_p, n_p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_exact(U, V, W):
        """Check if U, V, W form an exact decomposition of T."""
        r = U.shape[1]
        T_rec = np.zeros((n_m, m_p, n_p))
        # Efficient reconstruction using einsum
        # T_rec[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # We can do this in one go if we reshape, but for small r, loop is fine.
        # Actually, np.einsum('as,bs,cs->abc', U, V, W) is very efficient.
        T_rec = np.einsum('as,bs,cs->abc', U, V, W, optimize=True)
        return np.allclose(T_rec, T, atol=1e-9)

    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Attempt to find a lower rank decomposition.
    # Given the difficulty, we try a very limited search for rank 71 and 70.
    # Using random integer matrices with entries in {-1, 0, 1}.
    
    # If we find a lower rank, we update best.
    # We cap the number of attempts to ensure we finish well before deadline.
    
    search_ranks = [schoolbook_rank - 1, schoolbook_rank - 2]
    attempts_per_rank = 50
    
    for target_rank in search_ranks:
        if time.time() > deadline - 10:
            break
            
        for _ in range(attempts_per_rank):
            if time.time() > deadline - 5:
                break
                
            # Random integer matrices
            # Using small integers to avoid overflow issues in intermediate steps
            # though we verify at the end.
            U_try = np.random.randint(-1, 2, size=(n_m, target_rank))
            V_try = np.random.randint(-1, 2, size=(m_p, target_rank))
            W_try = np.random.randint(-1, 2, size=(n_p, target_rank))
            
            if verify_exact(U_try, V_try, W_try):
                best_rank = target_rank
                best_U = U_try.copy()
                best_V = V_try.copy()
                best_W = W_try.copy()
                break # Found for this rank, try lower
        
        if best_rank < schoolbook_rank - 1:
            break # Found a rank lower than the last target, stop searching further down

    # Convert to lists for return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()
    
    return {"U": U_ret, "V": V_ret, "W": W_ret}
