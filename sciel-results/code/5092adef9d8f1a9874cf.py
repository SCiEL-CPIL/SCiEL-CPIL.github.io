"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np
import scipy.optimize as opt

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _als_decompose(T, rank, n, m, p, max_iters=1000, tol=1e-12, init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: target tensor (n*m, m*p, p*n)
    rank: target rank
    Returns U, V, W if converged, else None.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    # Normalize initial vectors for stability
    for r in range(rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw
            
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
            
        # Update U
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                return None
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    return None

def _polish(U, V, W, T, n, m, p, deadline):
    """
    Use scipy.optimize.least_squares to polish the solution.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Flatten parameters
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def residual(x):
        U_curr = x[:dim_a*rank].reshape(dim_a, rank)
        V_curr = x[dim_a*rank:dim_a*rank+dim_b*rank].reshape(dim_b, rank)
        W_curr = x[dim_a*rank+dim_b*rank:].reshape(dim_c, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
        return (T_approx - T).flatten()
    
    # Check deadline
    if time.time() > deadline:
        return None
        
    try:
        # Use 'trf' (Trust Region Reflective) which is robust for large problems
        result = opt.least_squares(residual, x0, method='trf', max_nfev=1000)
        
        if result.success or result.cost < 1e-10:
            x_opt = result.x
            U_opt = x_opt[:dim_a*rank].reshape(dim_a, rank)
            V_opt = x_opt[dim_a*rank:dim_a*rank+dim_b*rank].reshape(dim_b, rank)
            W_opt = x_opt[dim_a*rank+dim_b*rank:].reshape(dim_c, rank)
            
            if _verify_fast(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                return U_opt, V_opt, W_opt
    except Exception:
        pass
        
    return None

def _try_rank(target_rank, n, m, p, deadline, max_attempts=200, seed=42):
    """
    Attempt to find a decomposition of given rank using ALS + Polish.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    n_drop = total_terms - target_rank
    
    if n_drop <= 0:
        return U0, V0, W0
    
    all_indices = list(range(total_terms))
    rng = random.Random(seed)
    
    best_error = float('inf')
    best_result = None
    
    # Strategy 1: Random initialization
    for attempt in range(max_attempts // 2):
        if time.time() > deadline:
            break
            
        init_U = np.random.randn(dim_a, target_rank)
        init_V = np.random.randn(dim_b, target_rank)
        init_W = np.random.randn(dim_c, target_rank)
        
        result = _als_decompose(T, target_rank, n, m, p, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        if result:
            U, V, W = result
            err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            if err < best_error:
                best_error = err
                best_result = (U, V, W)
            if err < 1e-7:
                # Try to polish
                polished = _polish(U, V, W, T, n, m, p, deadline)
                if polished:
                    return polished
                return U, V, W

    # Strategy 2: Schoolbook-based initialization (drop random columns)
    for attempt in range(max_attempts // 2):
        if time.time() > deadline:
            break
            
        drop_indices = rng.sample(all_indices, n_drop)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices].copy()
        init_V = V0[:, keep_indices].copy()
        init_W = W0[:, keep_indices].copy()
        
        # Add small random perturbation
        perturb_scale = rng.uniform(0.1, 0.5)
        for r_idx in range(len(keep_indices)):
            init_U[:, r_idx] += rng.uniform(-perturb_scale, perturb_scale, size=dim_a)
            init_V[:, r_idx] += rng.uniform(-perturb_scale, perturb_scale, size=dim_b)
            init_W[:, r_idx] += rng.uniform(-perturb_scale, perturb_scale, size=dim_c)
            
        result = _als_decompose(T, target_rank, n, m, p, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        if result:
            U, V, W = result
            err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            if err < best_error:
                best_error = err
                best_result = (U, V, W)
            if err < 1e-7:
                polished = _polish(U, V, W, T, n, m, p, deadline)
                if polished:
                    return polished
                return U, V, W

    # If we have a good candidate but not exact, try to polish the best one
    if best_result and best_error < 1e-4:
        U, V, W = best_result
        polished = _polish(U, V, W, T, n, m, p, deadline)
        if polished:
            return polished
        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
            return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try ranks in order of increasing difficulty
    # Known minimum rank is 23
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline, max_attempts=100, seed=42)
        if result is not None:
            U, V, W = result
            # Optional: Snap to nearest integer/half-integer if very close
            # This can help if the evaluator is strict about types, though float is usually fine
            # if the values are exact.
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
