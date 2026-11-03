import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    return T_target

def _verify(U, V, W, n, m, p, T_target=None, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if T_target is None:
        T_target = _build_target_tensor(n, m, p)
    
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _objective(params, T_target, n, m, p, rank):
    """
    Objective function for least_squares.
    params: flattened [U, V, W]
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    size_U = dim_U * rank
    size_V = dim_V * rank
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(dim_U, rank)
    V = V_flat.reshape(dim_V, rank)
    W = W_flat.reshape(dim_W, rank)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    # Return the residual vector
    return (T_recon - T_target).flatten()

def _als_decompose(T_target, n, m, p, rank, max_iter=200, tol=1e-12, rng=None, init=None):
    """
    Alternating Least Squares for tensor decomposition.
    Minimizes ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    if rng is None:
        rng = np.random.RandomState()
    
    if init is not None:
        U, V, W = init
    else:
        U = rng.randn(dim_U, rank) * 0.5
        V = rng.randn(dim_V, rank) * 0.5
        W = rng.randn(dim_W, rank) * 0.5
    
    prev_cost = np.inf
    
    for it in range(max_iter):
        # Solve for U: fix V, W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
        # Reshape: for each a, we have a system in U[a,:]
        # T_a = U[a,:] @ (V^T * W^T) where * is element-wise per column
        # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let M_U[r, b, c] = V[b,r] * W[c,r]
        # Then T[a] = sum_r U[a,r] * M_U[r]
        # In matrix form: T_a (flattened) = U[a,:] @ M_U_flat
        # where M_U_flat has shape (rank, dim_V * dim_W)
        
        # More efficiently:
        # For each r, the contribution to T from rank-r is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Fixing V,W: T = sum_r U[:,r] ⊗ (V[:,r] * W[:,r] reshaped)
        # Let's use: T.reshape(dim_U, dim_V*dim_W) = U @ (V*W).T reshaped
        
        # Compute VW = V ⊙ W element-wise: VW[r, b, c] = V[b,r] * W[c,r]
        VW = np.einsum('br,cr->bcr', V, W)  # shape (dim_V, dim_W, rank)
        VW_flat = VW.transpose(2, 0, 1).reshape(rank, dim_V * dim_W)  # shape (rank, dim_V*dim_W)
        
        T_flat = T_target.reshape(dim_U, dim_V * dim_W)
        
        # Solve: T_flat = U @ VW_flat
        # U = T_flat @ VW_flat^+ (pseudo-inverse)
        U_new, _, _, _ = np.linalg.lstsq(VW_flat.T, T_flat.T, rcond=None)
        U = U_new.T  # shape (dim_U, rank)
        
        # Solve for V: fix U, W
        UW = np.einsum('ar,cr->acr', U, W)  # shape (dim_U, dim_W, rank)
        UW_flat = UW.transpose(2, 0, 1).reshape(rank, dim_U * dim_W)
        
        T_flat2 = T_target.reshape(dim_U * dim_W, dim_V)
        V_new, _, _, _ = np.linalg.lstsq(UW_flat.T, T_flat2.T, rcond=None)
        V = V_new.T  # shape (dim_V, rank)
        
        # Solve for W: fix U, V
        UV = np.einsum('ar,br->abr', U, V)  # shape (dim_U, dim_V, rank)
        UV_flat = UV.transpose(2, 0, 1).reshape(rank, dim_U * dim_V)
        
        T_flat3 = T_target.reshape(dim_U * dim_V, dim_W)
        W_new, _, _, _ = np.linalg.lstsq(UV_flat.T, T_flat3.T, rcond=None)
        W = W_new.T  # shape (dim_W, rank)
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        cost = np.sum((T_recon - T_target)**2)
        
        if abs(prev_cost - cost) < tol * max(1.0, cost):
            break
        prev_cost = cost
    
    return U, V, W, cost

def _try_rank_opt(rank, n, m, p, max_time=10, seed=42):
    """Try to find a rank-rank decomposition via least_squares and ALS."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor
    T_target = _build_target_tensor(n, m, p)

    total_vars = (dim_U + dim_V + dim_W) * rank
    
    rng = np.random.RandomState(seed)
    
    # Try a few restarts
    for restart in range(8):
        if time.time() > _DEADLINE - 2:
            break
            
        # Method 1: ALS
        try:
            U_als, V_als, W_als, cost_als = _als_decompose(
                T_target, n, m, p, rank, 
                max_iter=300, tol=1e-14, rng=rng
            )
            
            if cost_als < 1e-8:
                # Try rounding strategies
                for mult in [1, 2, 4]:
                    U_rounded = np.round(U_als * mult) / mult
                    V_rounded = np.round(V_als * mult) / mult
                    W_rounded = np.round(W_als * mult) / mult
                    
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, T_target):
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    
                # Try integer rounding
                U_int = np.round(U_als)
                V_int = np.round(V_als)
                W_int = np.round(W_als)
                if _verify(U_int, V_int, W_int, n, m, p, T_target):
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
        except Exception as e:
            pass
        
        if time.time() > _DEADLINE - 2:
            break
        
        # Method 2: Levenberg-Marquardt
        try:
            # Initialize with small random values
            x0 = rng.randn(total_vars) * 0.5
            
            # Use least_squares
            result = scipy.optimize.least_squares(
                _objective, 
                x0, 
                args=(T_target, n, m, p, rank),
                method='lm', # Levenberg-Marquardt is good for small problems
                max_nfev=3000
            )
            
            if result.success or result.cost < 1e-8:
                # Extract U, V, W
                size_U = dim_U * rank
                size_V = dim_V * rank
                
                U_flat = result.x[:size_U]
                V_flat = result.x[size_U:size_U + size_V]
                W_flat = result.x[size_U + size_V:]
                
                U = U_flat.reshape(dim_U, rank)
                V = V_flat.reshape(dim_V, rank)
                W = W_flat.reshape(dim_W, rank)
                
                # Try rounding strategies
                for mult in [1, 2, 4]:
                    U_rounded = np.round(U * mult) / mult
                    V_rounded = np.round(V * mult) / mult
                    W_rounded = np.round(W * mult) / mult
                    
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, T_target):
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    
                # Try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify(U_int, V_int, W_int, n, m, p, T_target):
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
        except Exception as e:
            pass
            
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 18 to 24
    # 18 is the known minimal rank for 2x3x4
    for rank in range(18, 25):
        if time.time() > _DEADLINE - 5:
            break
            
        # Try multiple seeds
        for seed in range(20):
            if time.time() > _DEADLINE - 2:
                break
            res = _try_rank_opt(rank, n, m, p, seed=seed)
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
