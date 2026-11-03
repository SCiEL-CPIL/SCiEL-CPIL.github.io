import time
import numpy as np

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T


def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2
    T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)


def _max_abs_residual(U, V, W, T):
    """Compute max absolute residual."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(reconstructed - T))


def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W


def _als_refine(U, V, W, T, Nu, Nv, Nw, max_iter=100, tol=1e-10, deadline=None):
    """
    Alternating Least Squares refinement.
    U: (Nu, R), V: (Nv, R), W: (Nw, R)
    T: (Nu, Nv, Nw)
    """
    R = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Update U
        # For each u: T[u, v, w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # Fix V, W. M_U[(v,w), r] = V[v,r] * W[w,r]
        # T[u, :, :].flatten() = M_U.T @ U[u, :].T
        M_U = (V[:, :, None] * W[None, :, :]).reshape(Nv * Nw, R)
        
        for u in range(Nu):
            b_u = T[u, :, :].flatten()
            try:
                U[u, :] = np.linalg.lstsq(M_U.T, b_u, rcond=None)[0]
            except:
                pass
        
        # Update V
        M_V = (U[:, :, None] * W[None, :, :]).reshape(Nu * Nw, R)
        
        for v in range(Nv):
            b_v = T[:, v, :].flatten()
            try:
                V[v, :] = np.linalg.lstsq(M_V.T, b_v, rcond=None)[0]
            except:
                pass
        
        # Update W
        M_W = (U[:, :, None] * V[None, :, :]).reshape(Nu * Nv, R)
        
        for w in range(Nw):
            b_w = T[:, :, w].flatten()
            try:
                W[w, :] = np.linalg.lstsq(M_W.T, b_w, rcond=None)[0]
            except:
                pass
        
        err = _max_abs_residual(U, V, W, T)
        if err < tol:
            break
            
    return U, V, W


def _try_rank(n, m, p, rank, T, Nu, Nv, Nw, seed=0, time_limit=3.0):
    """
    Attempt to find a decomposition of the given rank.
    Returns (U, V, W, error) if successful, or None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize randomly with normalization
    U = rng.randn(Nu, rank)
    V = rng.randn(Nv, rank)
    W = rng.randn(Nw, rank)
    
    # Normalize columns to prevent divergence
    for r in range(rank):
        max_u = np.max(np.abs(U[:, r]))
        max_v = np.max(np.abs(V[:, r]))
        max_w = np.max(np.abs(W[:, r]))
        if max_u > 0:
            U[:, r] /= max_u
        if max_v > 0:
            V[:, r] /= max_v
        if max_w > 0:
            W[:, r] /= max_w
    
    start_time = time.time()
    
    # Run ALS
    U, V, W = _als_refine(U, V, W, T, Nu, Nv, Nw, max_iter=50, tol=1e-9, deadline=time.time() + time_limit)
    
    err = _max_abs_residual(U, V, W, T)
    
    if err < 1e-8:
        # Found a solution! Do a polish pass with tighter tolerance
        U, V, W = _als_refine(U, V, W, T, Nu, Nv, Nw, max_iter=100, tol=1e-12, deadline=time.time() + 1.0)
        err = _max_abs_residual(U, V, W, T)
        if err < 1e-8:
            return U, V, W, err
    
    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    Nu = n * m  # 10
    Nv = m * p  # 25
    Nw = p * n  # 10
    
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook_solution(n, m, p)
    U_best, V_best, W_best = U_fb.copy(), V_fb.copy(), W_fb.copy()
    best_rank = n * m * p  # 50
    
    # Known lower bound for 2x5x5 is around 40-42.
    # Try ranks from low to high, starting with the most promising.
    # The true minimal rank is likely in [40, 50].
    candidate_ranks = list(range(40, 51))
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        # Use more seeds for lower ranks where solutions are harder
        num_seeds = 30 if rank <= 42 else 15
        
        found = False
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            # Time limit per seed: use remaining time divided by remaining seeds
            remaining_seeds = num_seeds - seed
            time_limit = min(3.0, (DEADLINE_SECONDS - 5 - (time.time() - start_time)) / max(remaining_seeds, 1))
            if time_limit < 0.5:
                break
                
            try:
                result = _try_rank(n, m, p, rank, T, Nu, Nv, Nw, seed=seed * 7 + 3, time_limit=time_limit)
                if result is not None:
                    U_new, V_new, W_new, err = result
                    # Verify with high precision
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-6:
                        U_best, V_best, W_best = U_new, V_new, W_new
                        best_rank = rank
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            # Found a solution at this rank. Since we're going from low to high,
            # this is the best (lowest) rank we've found so far.
            # We can continue to see if even lower rank works, but since we're
            # going ascending, we should break and return this.
            # Actually, we're going from 40 upward, so if 40 works, that's great.
            # If 40 fails, try 41, etc. Once we find one, that's our answer.
            break
    
    # Convert to lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
