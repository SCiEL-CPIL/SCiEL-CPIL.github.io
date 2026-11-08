"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification using numpy."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # This is a tensor contraction.
        # We can compute this by iterating over i, j, k for small sizes.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol

def _local_search_discrete(U, V, W, n, m, p, max_iter=1000, deadline=None):
    """
    Local search on discrete values {-1, 0, 1} to minimize error.
    Returns (U, V, W, err)
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    def compute_error():
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        return np.sum((T - 1.0)**2)

    best_err = compute_error()
    if best_err < 1e-10:
        return U, V, W, best_err

    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Pick a random matrix and a random entry
        mat_idx = rng.integers(3)
        if mat_idx == 0:
            mat = U
            rows, cols = n * m, rank
        elif mat_idx == 1:
            mat = V
            rows, cols = m * p, rank
        else:
            mat = W
            rows, cols = p * n, rank
            
        r = rng.integers(rows)
        c = rng.integers(cols)
        
        old_val = mat[r, c]
        # Try changing to -1, 0, 1 (excluding old_val)
        candidates = [-1, 0, 1]
        candidates.remove(old_val)
        # Add old_val back if we want to allow no-change? No, we must change.
        # If all candidates are same as old_val (impossible with 3 unique values), skip.
        
        new_err = float('inf')
        best_new_val = old_val
        
        for val in candidates:
            mat[r, c] = val
            err = compute_error()
            if err < new_err:
                new_err = err
                best_new_val = val
        
        mat[r, c] = best_new_val
        
        if new_err < best_err:
            best_err = new_err
            if best_err < 1e-10:
                break
        else:
            # Revert if it got worse? The local search usually keeps the best.
            # Here we kept the best candidate. If best candidate is worse than global best, 
            # we might want to revert to global best state? 
            # For simplicity, we accept the local improvement. If it degrades, we might get stuck.
            # Let's implement a simple backtracking: if new_err > best_err, revert to old_val.
            if new_err > best_err:
                mat[r, c] = old_val

    return U, V, W, best_err

def _try_rank_discrete(rank, n, m, p, deadline, num_restarts=5):
    """Try to find a valid decomposition of given rank using discrete local search."""
    best_U, best_V, best_W, best_err = None, None, None, float('inf')
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        # Random initialization with {-1, 0, 1}
        np.random.seed(100 + restart)
        U = np.random.choice([-1, 0, 1], size=(n * m, rank))
        V = np.random.choice([-1, 0, 1], size=(m * p, rank))
        W = np.random.choice([-1, 0, 1], size=(p * n, rank))
        
        # Scale? No, discrete search.
        U, V, W, err = _local_search_discrete(U, V, W, n, m, p, max_iter=500, deadline=deadline)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try lower ranks in increasing order
    # Lower bound: max(n*m, m*p, p*n) = 15
    # Try 14, 15, 16
    for rank in range(14, best_rank + 1):
        if time.time() > deadline - 20:
            break
        # Only try discrete search for small ranks
        if rank <= 16:
            result = _try_rank_discrete(rank, n, m, p, deadline, num_restarts=3)
            if result is not None:
                U, V, W = result
                if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break  # Found a valid lower rank, stop
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
