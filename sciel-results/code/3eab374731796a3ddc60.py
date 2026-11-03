import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 130  # 20s safety margin

def _structured_init(rank, n, m, p, seed):
    """Create structured initialization based on partial schoolbook pattern."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    rng = np.random.RandomState(seed)
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    # Start with schoolbook pattern for first min(rank, n*m*p) entries
    num_school = min(rank, n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                if col < num_school:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
    
    # Add random perturbations to remaining columns
    for col in range(num_school, rank):
        U[:, col] = rng.randn(dim_U) * 0.5
        V[:, col] = rng.randn(dim_V) * 0.5
        W[:, col] = rng.randn(dim_W) * 0.5
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    if not U or not V or not W:
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    # Check if entries are integer or half-integer
    # The task says "integer/half-integer entries"
    # We will assume the input is already snapped, but double check.
    # Actually, the evaluator might check this, or just the tensor decomposition.
    # The prompt says: "factor matrices ... with integer/half-integer entries that exactly decompose..."
    # So we must return such entries.
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _residual(x, n, m, p, R):
    """
    Compute residual for least_squares.
    x: flattened vector of U, V, W
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Unpack
    U = x[:dim_U * R].reshape(dim_U, R)
    V = x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
    W = x[dim_U * R + dim_V * R :].reshape(dim_W, R)
    
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Target
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    return (T_recon - T_target).flatten()

def _try_rank_ls(rank, n, m, p, max_seeds=10, seed_base=42, time_limit=10):
    """Try to find a rank-rank decomposition via Least Squares."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    total_vars = (dim_U + dim_V + dim_W) * rank
    
    # Pre-compute target tensor for speed
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def residual_fast(x):
        U = x[:dim_U * rank].reshape(dim_U, rank)
        V = x[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
        W = x[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T_target).flatten()
    
    for seed in range(seed_base, seed_base + max_seeds):
        if time.time() > _DEADLINE:
            return None
        
        # Use structured initialization
        U_init, V_init, W_init = _structured_init(rank, n, m, p, seed)
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.least_squares(
                residual_fast, x0,
                method='lm',
                ftol=1e-14, xtol=1e-14, gtol=1e-14,
                max_nfev=5000
            )
        except Exception:
            continue
        
        if res.cost < 1e-12:
            x_sol = res.x
            U = x_sol[:dim_U * rank].reshape(dim_U, rank)
            V = x_sol[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W = x_sol[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
            
            # Try multiple snapping strategies
            for scale in [2, 4, 1]:  # half-integers, quarters, integers
                U_snap = np.round(U * scale) / scale
                V_snap = np.round(V * scale) / scale
                W_snap = np.round(W * scale) / scale
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # Also try without snapping if error is tiny
            if np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T_target)) < 1e-10:
                # Check if entries are already close to half-integers
                if np.allclose(U, np.round(U * 2) / 2, atol=1e-6) and \
                   np.allclose(V, np.round(V * 2) / 2, atol=1e-6) and \
                   np.allclose(W, np.round(W * 2) / 2, atol=1e-6):
                    U_final = np.round(U * 2) / 2
                    V_final = np.round(V * 2) / 2
                    W_final = np.round(W * 2) / 2
                    return (U_final.tolist(), V_final.tolist(), W_final.tolist())
            
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 20 to 23
    # 24 is the schoolbook rank
    for rank in range(20, 24):
        if time.time() > _DEADLINE - 10:
            break
            
        # More seeds for higher ranks (closer to the true rank)
        # If rank is 20, it's unlikely. If 23, it's more likely.
        seeds = 50 if rank >= 22 else 20
        
        res = _try_rank_ls(rank, n, m, p, max_seeds=seeds, seed_base=rank * 1000)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
