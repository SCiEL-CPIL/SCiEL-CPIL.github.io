"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

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

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication target tensor."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = _build_target_tensor(n, m, p)
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _als_decompose(T, rank, n, m, p, max_iters=500, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Alternating Least Squares for tensor decomposition with adaptive step sizes.
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
    stall_count = 0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            # Return best found so far if it's good enough
            if best_error < 1e-6:
                return best_U, best_V, best_W
            return None
            
        # Update U: M[bc, r] = V[b,r] * W[c,r], computed ONCE
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V: M[ac, r] = U[a,r] * W[c,r], computed ONCE
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W: M[ab, r] = U[a,r] * V[b,r], computed ONCE
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        # Track best
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if abs(prev_error - error) < tol:
            stall_count += 1
            if stall_count >= 3:
                if error < 1e-7:
                    return U, V, W
                else:
                    # Try a polish pass with tighter tolerance
                    return _polish_pass(U, V, W, T, rank, n, m, p, deadline)
        else:
            stall_count = 0
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    # Try polish if close
    if error < 1e-4:
        return _polish_pass(U, V, W, T, rank, n, m, p, deadline)
    return None

def _polish_pass(U, V, W, T, rank, n, m, p, deadline, max_iters=200, tol=1e-12):
    """High-precision polish pass using tighter tolerance and more iterations."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
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
            if error < 1e-8:
                return U, V, W
            else:
                return None
        prev_error = error
        
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-8:
        return U, V, W
    return None

def _generate_init_strategies(U0, V0, W0, target_rank, n, m, p, rng):
    """Generate multiple diverse initialization strategies."""
    total_terms = n * m * p
    n_drop = total_terms - target_rank
    all_indices = list(range(total_terms))
    strategies = []
    
    # Strategy 1: Random drop with small perturbation
    drop_indices = rng.sample(all_indices, n_drop)
    keep_indices = [i for i in all_indices if i not in drop_indices]
    init_U = U0[:, keep_indices].copy()
    init_V = V0[:, keep_indices].copy()
    init_W = W0[:, keep_indices].copy()
    perturb_scale = rng.uniform(0.02, 0.15)
    for r_idx in range(len(keep_indices)):
        mask_u = init_U[:, r_idx] != 0
        mask_v = init_V[:, r_idx] != 0
        mask_w = init_W[:, r_idx] != 0
        if mask_u.any():
            init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
        if mask_v.any():
            init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
        if mask_w.any():
            init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
    strategies.append((init_U, init_V, init_W))
    
    # Strategy 2: Random drop with larger perturbation
    drop_indices = rng.sample(all_indices, n_drop)
    keep_indices = [i for i in all_indices if i not in drop_indices]
    init_U = U0[:, keep_indices].copy()
    init_V = V0[:, keep_indices].copy()
    init_W = W0[:, keep_indices].copy()
    perturb_scale = rng.uniform(0.1, 0.5)
    for r_idx in range(len(keep_indices)):
        mask_u = init_U[:, r_idx] != 0
        mask_v = init_V[:, r_idx] != 0
        mask_w = init_W[:, r_idx] != 0
        if mask_u.any():
            init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
        if mask_v.any():
            init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
        if mask_w.any():
            init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
    strategies.append((init_U, init_V, init_W))
    
    # Strategy 3: Random initialization with structure
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    init_U = np.zeros((dim_a, target_rank))
    init_V = np.zeros((dim_b, target_rank))
    init_W = np.zeros((dim_c, target_rank))
    for r_idx in range(target_rank):
        # Each factor gets some structure
        for a in range(dim_a):
            if rng.random() < 0.3:
                init_U[a, r_idx] = rng.uniform(-1, 1)
        for b in range(dim_b):
            if rng.random() < 0.3:
                init_V[b, r_idx] = rng.uniform(-1, 1)
        for c in range(dim_c):
            if rng.random() < 0.3:
                init_W[c, r_idx] = rng.uniform(-1, 1)
    strategies.append((init_U, init_V, init_W))
    
    # Strategy 4: Sparse random with more density
    init_U = np.zeros((dim_a, target_rank))
    init_V = np.zeros((dim_b, target_rank))
    init_W = np.zeros((dim_c, target_rank))
    for r_idx in range(target_rank):
        for a in range(dim_a):
            if rng.random() < 0.5:
                init_U[a, r_idx] = rng.uniform(-1, 1)
        for b in range(dim_b):
            if rng.random() < 0.5:
                init_V[b, r_idx] = rng.uniform(-1, 1)
        for c in range(dim_c):
            if rng.random() < 0.5:
                init_W[c, r_idx] = rng.uniform(-1, 1)
    strategies.append((init_U, init_V, init_W))
    
    return strategies

def _try_rank(target_rank, n, m, p, deadline, max_attempts=100, seed=42):
    """
    Attempt to find a decomposition of given rank using ALS with multiple strategies.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _build_target_tensor(n, m, p)
    
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    n_drop = total_terms - target_rank
    
    if n_drop <= 0:
        return U0, V0, W0
    
    rng = random.Random(seed)
    
    # Generate multiple initialization strategies
    strategies = _generate_init_strategies(U0, V0, W0, target_rank, n, m, p, rng)
    
    # Try each strategy multiple times with different perturbations
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Cycle through strategies
        strategy_idx = attempt % len(strategies)
        init_U, init_V, init_W = strategies[strategy_idx]
        
        # Add additional perturbation for diversity
        perturb_scale = rng.uniform(0.01, 0.2)
        init_U = init_U.copy()
        init_V = init_V.copy()
        init_W = init_W.copy()
        for r_idx in range(target_rank):
            mask_u = init_U[:, r_idx] != 0
            mask_v = init_V[:, r_idx] != 0
            mask_w = init_W[:, r_idx] != 0
            if mask_u.any():
                init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_v.any():
                init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_w.any():
                init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            
        # Run ALS
        result = _als_decompose(T, target_rank, n, m, p, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Try ranks in order of increasing difficulty
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline, max_attempts=80, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
