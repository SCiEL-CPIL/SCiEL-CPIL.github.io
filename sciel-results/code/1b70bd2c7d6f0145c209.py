import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin for 150s limit

def _build_target_tensor(n, m, p):
    """Construct the target matrix multiplication tensor."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, T_target, tol=1e-9):
    """
    Verify if U, V, W form a valid decomposition of T_target.
    Also checks if entries are close to integers or half-integers.
    """
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False, None, None, None
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False, None, None, None
        
    # Check if entries are integer or half-integer
    # The problem states "integer/half-integer entries".
    # We check if 2*U, 2*V, 2*W are close to integers.
    if not (np.allclose(U * 2, np.round(U * 2), atol=1e-6) and
            np.allclose(V * 2, np.round(V * 2), atol=1e-6) and
            np.allclose(W * 2, np.round(W * 2), atol=1e-6)):
        return False, None, None, None

    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Exact check (using float is risky for 'exact', but we check residual)
    T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False, None, None, None
        
    return True, U_snap, V_snap, W_snap

def _try_rank(rank, n, m, p, T_target, max_seeds=50, seed_base=1000):
    """
    Attempt to find a decomposition of the given rank.
    Uses Levenberg-Marquardt with multiple initializations.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Total variables: (dim_U + dim_V + dim_W) * rank
    # Total residuals: dim_U * dim_V * dim_W
    
    def residual(x):
        U = x[:dim_U * rank].reshape(dim_U, rank)
        V = x[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
        W = x[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T_target).flatten()
    
    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        rng = np.random.RandomState(seed_base + seed)
        
        # Scale of init matters. 
        # The entries of T are 1. 
        # If U, V, W are ~ N(0, 1), then T_recon entries are sums of products.
        # We want T_recon ~ 1.
        
        U_init = rng.randn(dim_U, rank) * 0.5
        V_init = rng.randn(dim_V, rank) * 0.5
        W_init = rng.randn(dim_W, rank) * 0.5
        
        # Add a small component of the schoolbook pattern to guide it?
        # Let's try mixing.
        if seed % 2 == 1:
             # Structured: take first 'rank' schoolbook columns and perturb
             U_init = np.zeros((dim_U, rank))
             V_init = np.zeros((dim_V, rank))
             W_init = np.zeros((dim_W, rank))
             
             # Fill with schoolbook pattern for 'rank' terms
             count = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         if count >= rank: break
                         U_init[i*m+j, count] = 1.0
                         V_init[j*p+k, count] = 1.0
                         W_init[k*n+i, count] = 1.0
                         count += 1
                         if count >= rank: break
                     if count >= rank: break
                 if count >= rank: break
             
             # Perturb
             U_init += rng.randn(dim_U, rank) * 0.1
             V_init += rng.randn(dim_V, rank) * 0.1
             W_init += rng.randn(dim_W, rank) * 0.1
             
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.least_squares(
                residual, x0,
                method='lm',
                ftol=1e-14, xtol=1e-14, gtol=1e-14,
                max_nfev=5000
            )
        except Exception:
            continue
            
        # Check for near-exact solution
        if res.cost < 1e-12:
            x_sol = res.x
            U = x_sol[:dim_U * rank].reshape(dim_U, rank)
            V = x_sol[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W = x_sol[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
            
            # Verify and snap
            result = _verify_exact(U, V, W, n, m, p, T_target)
            if result[0]:
                return result[1], result[2], result[3]
                
    return None

def _get_schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = 2, 3, 4
    
    T_target = _build_target_tensor(n, m, p)
    
    # Try ranks from 18 upwards. Rank 18 is the known minimum for 2x3x4.
    # We use a tighter deadline to ensure we have time to return a valid result.
    for rank in [18, 19, 20, 21, 22, 23]:
        if time.time() > _DEADLINE - 10:  # Reserve 10s for fallback and safety
            break
        # Use more seeds for lower ranks as they are harder
        max_seeds = 150 if rank <= 18 else 75
        res = _try_rank(rank, n, m, p, T_target, max_seeds=max_seeds, seed_base=42 + rank)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook (Rank 24)
    U, V, W = _get_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
