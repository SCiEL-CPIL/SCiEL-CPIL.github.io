import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 145  # 5s safety margin for 150s limit

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

def _als_solve(U, V, W, rank, n, m, p, T_target, max_iter=100):
    """
    Perform Alternating Least Squares to refine U, V, W.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Reshape T for matrix multiplication in ALS
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            return U, V, W, False
            
        # Solve for U:
        # T[a, b, c] = U[a, r] * (V[b, r] * W[c, r])
        # For fixed a, the equation is T[a, :, :] = U[a, :] @ (V * W)^T ? No.
        # Let M_VW[b, c] = sum_r V[b, r] W[c, r].
        # Then T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
        # This is not a simple matrix product because U depends on a, and V, W on b, c.
        # We can view it as:
        # T_reshaped[a, (b, c)] = U[a, r] * (V[b, r] * W[c, r])_reshaped
        # Let X_U[a, r] = U[a, r].
        # Let Y_U[b, c, r] = V[b, r] * W[c, r].
        # We want to minimize || T[a, b, c] - sum_r U[a, r] Y_U[b, c, r] ||^2.
        # This is a linear least squares problem for each a?
        # No, U is the variable.
        # Flatten (b, c) into index k.
        # T_flat[a, k] = sum_r U[a, r] * (V_flat[b(r), k] * W_flat[c(r), k])? No.
        
        # Standard ALS for 3-tensor:
        # Fix V, W.
        # For each a, we have T[a, :, :] = sum_r U[a, r] (V[:, r] * W[:, r])^T ? No.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
        # Let Z_VW[b, c] = V[b, :] * W[c, :] (element-wise product of vectors of length rank).
        # Then T[a, b, c] = U[a, :] . Z_VW[b, c] (dot product).
        # So for each a, T[a, :, :] is a matrix of size dim_V x dim_W.
        # And U[a, :] is a vector of size rank.
        # This is NOT a standard matrix multiplication.
        # It's a multilinear regression.
        
        # We can solve this by flattening b and c.
        # T_flat[a, b*dim_W + c] = sum_r U[a, r] * (V[b, r] * W[c, r]).
        # Let K = dim_V * dim_W.
        # Let M_U be a matrix of size K x rank.
        # M_U[b*dim_W + c, r] = V[b, r] * W[c, r].
        # Then T_flat[a, :] = U[a, :] @ M_U.
        # So U[a, :] = argmin || T_flat[a, :] - U[a, :] M_U ||.
        # This is a linear least squares problem: U[a, :] M_U = T_flat[a, :].
        # U[a, :] = T_flat[a, :] @ pinv(M_U).
        
        # Construct M_U
        # M_U[k, r] where k = b * dim_W + c
        # M_U[k, r] = V[b, r] * W[c, r]
        
        M_U = np.zeros((dim_V * dim_W, rank))
        for b in range(dim_V):
            for c in range(dim_W):
                k = b * dim_W + c
                M_U[k, :] = V[b, :] * W[c, :]
        
        # T_flat is dim_U x (dim_V * dim_W)
        T_flat = T_target.reshape(dim_U, dim_V * dim_W)
        
        # Solve U
        # U[a, :] M_U = T_flat[a, :]
        # This is underdetermined if rank < dim_V*dim_W?
        # We have rank variables for U[a, :].
        # We have dim_V*dim_W equations.
        # Least Squares: U[a, :] = T_flat[a, :] @ pinv(M_U)
        
        try:
            U_new = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)[0].T
        except:
            U_new = U
            
        # Solve V
        # T[a, b, c] = sum_r V[b, r] U[a, r] W[c, r]
        # M_V[a, c, r] = U[a, r] * W[c, r]
        # T_flat_V[b, a*dim_W + c] = sum_r V[b, r] M_V[a, c, r]
        
        M_V = np.zeros((dim_U * dim_W, rank))
        for a in range(dim_U):
            for c in range(dim_W):
                k = a * dim_W + c
                M_V[k, :] = U_new[a, :] * W[c, :]
                
        T_flat_V = T_target.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W)
        
        try:
            V_new = np.linalg.lstsq(M_V.T, T_flat_V.T, rcond=None)[0].T
        except:
            V_new = V
            
        # Solve W
        # T[a, b, c] = sum_r W[c, r] U[a, r] V[b, r]
        # M_W[b, a, r] = U[a, r] * V[b, r]
        # T_flat_W[c, b*dim_U + a] = sum_r W[c, r] M_W[b, a, r]
        
        M_W = np.zeros((dim_U * dim_V, rank))
        for a in range(dim_U):
            for b in range(dim_V):
                k = b * dim_U + a
                M_W[k, :] = U_new[a, :] * V_new[b, :]
                
        T_flat_W = T_target.transpose(2, 1, 0).reshape(dim_W, dim_U * dim_V)
        
        try:
            W_new = np.linalg.lstsq(M_W.T, T_flat_W.T, rcond=None)[0].T
        except:
            W_new = W
            
        U, V, W = U_new, V_new, W_new
        
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.max(np.abs(T_recon - T_target))
    return U, V, W, error < 1e-8

def _try_rank(rank, n, m, p, T_target, max_seeds=10, seed_base=1000):
    """
    Attempt to find a decomposition of the given rank using ALS.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        rng = np.random.RandomState(seed_base + seed)
        
        # Initialize with integers
        # Use small integers to help convergence to integer solutions
        U_init = rng.randint(-2, 3, size=(dim_U, rank)).astype(float)
        V_init = rng.randint(-2, 3, size=(dim_V, rank)).astype(float)
        W_init = rng.randint(-2, 3, size=(dim_W, rank)).astype(float)
        
        # Run ALS
        U_sol, V_sol, W_sol, converged = _als_solve(U_init, V_init, W_init, rank, n, m, p, T_target)
        
        if converged:
            result = _verify_exact(U_sol, V_sol, W_sol, n, m, p, T_target)
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
    
    # Try rank 23
    res = _try_rank(23, n, m, p, T_target, max_seeds=20, seed_base=42)
    
    if res is not None:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook (Rank 24)
    U, V, W = _get_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
