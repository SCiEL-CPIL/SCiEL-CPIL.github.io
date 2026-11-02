"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _random_sparse_init(n, m, p, rank, density=0.1, seed=None):
    """
    Initialize U, V, W with sparse random integer entries.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Determine number of non-zeros per row
    # Heuristic: keep it sparse
    nnz_u = max(1, int(rank * density))
    nnz_v = max(1, int(rank * density))
    nnz_w = max(1, int(rank * density))
    
    for i in range(n * m):
        idx = rng.choice(rank, size=nnz_u, replace=False)
        U[i, idx] = rng.choice([-1, 1], size=nnz_u)
        
    for j in range(m * p):
        idx = rng.choice(rank, size=nnz_v, replace=False)
        V[j, idx] = rng.choice([-1, 1], size=nnz_v)
        
    for k in range(p * n):
        idx = rng.choice(rank, size=nnz_w, replace=False)
        W[k, idx] = rng.choice([-1, 1], size=nnz_w)
        
    return U, V, W

def _optimize_local(U, V, W, n, m, p, deadline):
    """
    Use Nelder-Mead to minimize the error of the decomposition.
    """
    U = U.copy().flatten()
    V = V.copy().flatten()
    W = W.copy().flatten()
    
    def error(params):
        U_flat, V_flat, W_flat = params[:U.size], params[U.size:U.size+V.size], params[U.size+V.size:]
        U_mat = U_flat.reshape(n*m, -1)
        V_mat = V_flat.reshape(m*p, -1)
        W_mat = W_flat.reshape(p*n, -1)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                    err += (s - 1.0)**2
        return err

    x0 = np.concatenate([U, V, W])
    
    # Use Nelder-Mead, but it can be slow. We limit iterations.
    # We only do this if we have time.
    try:
        from scipy.optimize import minimize
        res = minimize(error, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success and res.fun < 1e-4:
            x_opt = res.x
            U_opt = x_opt[:U.size].reshape(n*m, -1)
            V_opt = x_opt[U.size:U.size+V.size].reshape(m*p, -1)
            W_opt = x_opt[U.size+V.size:].reshape(p*n, -1)
            
            # Round to half-integers
            U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U_opt])
            V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V_opt])
            W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W_opt])
            
            return U_rounded, V_rounded, W_rounded
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin for 150s limit
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Search for lower ranks
    # We try ranks from 20 up to 79.
    # If we find a valid one, we update best_rank and continue to lower ranks?
    # No, we want the MINIMAL rank.
    # So we should search from low to high?
    # If we find rank 25, we must check 24, 23...
    # But if we search low to high, we stop at the first valid one?
    # Yes, because we want the minimum.
    
    # However, low ranks are harder to find.
    # High ranks are easier.
    # Strategy: Search high to low to get a quick improvement, then refine?
    # Or search low to high to guarantee minimality?
    # Given the time limit, finding *any* improvement is valuable.
    # Let's try a range of ranks.
    
    # We will iterate ranks from 79 down to 20.
    # For each rank, we try a few random initializations.
    # If we find a valid one, we record it.
    # We continue to lower ranks to see if we can go lower.
    # We stop if we run out of time.
    
    ranks_to_try = list(range(79, 19, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try a few seeds
        for seed in range(20):
            if time.time() > deadline - 5:
                break
                
            U_init, V_init, W_init = _random_sparse_init(n, m, p, rank, density=0.2, seed=seed)
            
            # Check if the random init is already valid (unlikely but possible for high rank)
            if _verify_decomposition(U_init.tolist(), V_init.tolist(), W_init.tolist(), n, m, p):
                best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
                best_rank = rank
                found = True
                break
                
            # Try local optimization
            # Only do this if we have enough time
            if time.time() < deadline - 10:
                res = _optimize_local(U_init, V_init, W_init, n, m, p, deadline)
                if res is not None:
                    U_r, V_r, W_r = res
                    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = rank
                        found = True
                        break
                        
        # If we found a solution for this rank, we can continue to lower ranks
        # to see if we can find an even better one.
        # But we must be careful not to spend too much time.
        
    return best_result
# EVOLVE-BLOCK-END
