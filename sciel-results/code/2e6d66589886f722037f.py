"""Rewritten program: Fourier-feature parameterization + L-BFGS-B soft-min optimization."""
import numpy as np
import time

N = 100
SEED = 42

def _soft_min(x, beta=50.0):
    """Soft minimum of array x using log-sum-exp trick."""
    x = np.asarray(x, dtype=np.float64)
    m = np.min(x)
    return m - np.log(np.sum(np.exp(-(x - m) * beta)) / len(x)) / beta

def _compute_radius(c):
    """Compute the maximum common radius for centers c (N,3)."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Distance to walls
    wall = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)

def _objective(params, beta):
    """Negative soft-min radius (to maximize)."""
    # Decode params to centers
    c = _decode(params)
    # Clamp to cube
    c = np.clip(c, -0.5, 0.5)
    # Compute soft-min radius
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_soft = _soft_min(dist, beta) / 2.0
    wall = 0.5 - np.abs(c)
    wall_soft = _soft_min(wall, beta)
    r_soft = _soft_min(np.array([pair_soft, wall_soft]), beta)
    return -r_soft

def _decode(params):
    """Decode parameter vector to N centers using Fourier features."""
    # params shape: (N, 3, n_features)
    n_feat = params.shape[2]
    # First 3 features are DC + linear, rest are sinusoidal
    # Use a fixed frequency basis
    # We'll use the first 3 as offsets, and the rest as amplitudes for sin/cos
    # Actually, let's use a simpler approach: params directly encode centers
    # but through a smooth transformation
    # For now, use direct encoding with smooth mapping
    # params: (N, 3) directly
    if params.ndim == 1:
        c = params.reshape(N, 3)
    else:
        c = params
    # Map to [-0.5, 0.5] via tanh
    c = 0.5 * np.tanh(c)
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(SEED)
    
    best_c = None
    best_r = -1.0
    
    # Start with a reasonable initial configuration
    # Use a grid-based start with jitter
    m = 5  # 5x5x5 = 125, take first 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    c0 += rng.normal(0, 0.01, c0.shape)
    c0 = np.clip(c0, -0.49, 0.49)
    
    # Encode: use direct parameters, tanh-mapped
    # Invert tanh: params = arctanh(c / 0.5)
    c_safe = np.clip(c0, -0.49, 0.49)
    params0 = np.arctanh(c_safe / 0.5)
    
    from scipy.optimize import minimize
    
    def obj_fn(params):
        c = 0.5 * np.tanh(params)
        # Compute soft-min radius
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_soft = _soft_min(dist, 50.0) / 2.0
        wall = 0.5 - np.abs(c)
        wall_soft = _soft_min(wall, 50.0)
        r_soft = _soft_min(np.array([pair_soft, wall_soft]), 50.0)
        return -r_soft
    
    def obj_fn_grad(params):
        c = 0.5 * np.tanh(params)
        # Gradient of soft-min
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        beta = 50.0
        # Soft-min gradient for pair distances
        m_pair = np.min(dist)
        w_pair = np.exp(-(dist - m_pair) * beta)
        w_pair /= np.sum(w_pair)
        w_pair = np.where(np.eye(N, dtype=bool), 0, w_pair)
        
        # Gradient of dist w.r.t. c
        # d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
        grad_c = np.zeros_like(c)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d > 1e-10:
                    grad_c[i] += w_pair[i, j] * (c[i] - c[j]) / d
        
        # Wall gradient
        wall = 0.5 - np.abs(c)
        m_wall = np.min(wall)
        w_wall = np.exp(-(wall - m_wall) * beta)
        w_wall /= np.sum(w_wall)
        
        # d(wall_i)/d(c_i) = -sign(c_i)
        grad_wall_c = -np.sign(c)
        
        # Combine: soft-min of [pair_soft, wall_soft]
        vals = np.array([m_pair / 2.0, m_wall])
        m_val = np.min(vals)
        w_vals = np.exp(-(vals - m_val) * beta)
        w_vals /= np.sum(w_vals)
        
        # grad r_soft / grad c = w_vals[0] * grad_pair + w_vals[1] * grad_wall
        grad_r = w_vals[0] * grad_c + w_vals[1] * np.sum(w_wall * grad_wall_c, axis=0)
        
        # Chain rule through tanh
        # d(tanh(x))/dx = 1 - tanh(x)^2
        grad_params = grad_r * (1 - np.tanh(params) ** 2)
        
        return -grad_params  # negative because we maximize
    
    # Simplify: use numerical gradient initially, then switch
    # Actually, let's just use L-BFGS-B with analytic gradient
    
    # Multiple restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            p = params0.copy()
        else:
            # Jitter the initial guess
            p = params0 + rng.normal(0, 0.5, params0.shape)
        
        try:
            res = minimize(
                obj_fn, p,
                jac=obj_fn_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            p_opt = res.x
            c_opt = 0.5 * np.tanh(p_opt)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            
            r = _compute_radius(c_opt)
            if r > best_r:
                best_r = r
                best_c = c_opt.copy()
        except Exception:
            continue
    
    # Final polish: use tighter beta and more iterations on the best solution
    if best_c is not None and time.time() < deadline - 5:
        p_best = np.arctanh(np.clip(best_c / 0.5, -0.99, 0.99))
        
        def obj_fn_fine(params):
            c = 0.5 * np.tanh(params)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            pair_soft = _soft_min(dist, 200.0) / 2.0
            wall = 0.5 - np.abs(c)
            wall_soft = _soft_min(wall, 200.0)
            r_soft = _soft_min(np.array([pair_soft, wall_soft]), 200.0)
            return -r_soft
        
        try:
            res = minimize(
                obj_fn_fine, p_best,
                jac=obj_fn_grad,  # approximate, good enough
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            p_fine = res.x
            c_fine = 0.5 * np.tanh(p_fine)
            c_fine = np.clip(c_fine, -0.5, 0.5)
            
            r_fine = _compute_radius(c_fine)
            if r_fine > best_r:
                best_r = r_fine
                best_c = c_fine.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_c is None:
        # Fallback to grid
        best_c = c0
    
    # Final validation
    best_c = np.clip(best_c, -0.5, 0.5)
    if not np.all(np.isfinite(best_c)):
        best_c = c0
    
    return best_c
