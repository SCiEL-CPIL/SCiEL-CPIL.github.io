import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def optimize_repulsion(centers, max_iter=1000, base_lr=0.05, decay=0.999):
    """Fast vectorized repulsion optimization."""
    centers = centers.copy().astype(float)
    lr = base_lr
    
    for it in range(max_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        boundary_dists = 0.5 - np.abs(centers)
        boundary_min = np.min(boundary_dists, axis=1)
        
        r_min = min(np.min(dists) / 2, np.min(boundary_min))
        if r_min <= 0:
            centers = np.clip(centers, -0.49, 0.49)
            continue
        
        # Pairwise forces (vectorized)
        push_mask = (dists < 2 * r_min) & ~np.eye(N, dtype=bool)
        mag_factor = np.zeros((N, N))
        mag_factor[push_mask] = (2 * r_min - dists[push_mask]) * 0.5 / dists[push_mask]
        forces = np.sum(mag_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        for i in range(N):
            for k in range(3):
                bd = boundary_dists[i, k]
                if bd < r_min:
                    sign = 1 if centers[i, k] > 0 else -1
                    forces[i, k] += sign * (r_min - bd)
        
        centers += lr * forces
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        lr *= decay
    
    return centers

def soft_min_objective(x, T=10.0):
    """Soft-min surrogate with analytical gradient for L-BFGS-B."""
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    bd_flat = boundary_dists.flatten()
    
    # All constraints
    all_c = np.concatenate([pair_dists, bd_flat])
    
    # Soft-min via log-sum-exp
    m = np.min(all_c)
    exp_terms = np.exp(-T * (all_c - m))
    soft_min = m - np.log(np.sum(exp_terms)) / T
    
    # Gradient
    # d(soft_min)/d(all_c) = -exp_terms / sum(exp_terms)
    weights = -exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    # For pair (i,j), constraint is |c_i - c_j|/2
    # d/d(c_i) = (c_i - c_j) / |c_i - c_j| * w_ij
    # d/d(c_j) = (c_j - c_i) / |c_i - c_j| * w_ij
    pair_weights = weights[:len(pair_dists)]
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = pair_weights[idx]
        d_ij = diff[i, j]
        dist_ij = dists[i, j]
        if dist_ij > 1e-15:
            grad[3*i:3*i+3] += w * d_ij / dist_ij
            grad[3*j:3*j+3] -= w * d_ij / dist_ij
    
    # Boundary part
    # constraint: 0.5 - |c_ik|
    # d/d(c_ik) = -sign(c_ik) * w
    bd_weights = weights[len(pair_dists):]
    bd_w_reshaped = bd_weights.reshape(N, 3)
    signs = np.sign(c)
    signs[c == 0] = 1.0
    grad -= bd_w_reshaped * signs
    
    return -soft_min, -grad

def polish_with_lbfgsb(centers, T=10.0, maxiter=200, ftol=1e-15):
    """High-precision polish using L-BFGS-B with soft-min surrogate."""
    def obj(x):
        val, _ = soft_min_objective(x, T)
        return val
    
    def grad(x):
        _, g = soft_min_objective(x, T)
        return g
    
    res = minimize(obj, centers.flatten(), jac=grad, method='L-BFGS-B',
                   bounds=[(-0.5, 0.5)] * (N * 3),
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-15})
    return res.x.reshape(N, 3)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_r = -np.inf
    best_centers = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Jittered 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Various random inits
    for seed in range(20):
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. Jittered grids with different jitter
    for jit in [0.02, 0.05, 0.1]:
        rng = np.random.RandomState(100 + int(jit*100))
        jittered = grid[:N].copy() + rng.uniform(-jit, jit, size=(N, 3))
        inits.append(np.clip(jittered, -0.49, 0.49))
    
    # Optimize each init
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        result = optimize_repulsion(init, max_iter=800, base_lr=0.05, decay=0.9995)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
        
        # If we have time, try scipy polish
        if time.time() < deadline - 30 and r > 0.13:
            try:
                # First polish with moderate T
                polished = polish_with_lbfgsb(result, T=10.0, maxiter=150, ftol=1e-14)
                r_polished = compute_score(polished)
                
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished.copy()
                
                # Second polish with tighter T for higher precision
                if time.time() < deadline - 15 and r_polished > 0.135:
                    polished2 = polish_with_lbfgsb(polished, T=30.0, maxiter=100, ftol=1e-15)
                    r_polished2 = compute_score(polished2)
                    if r_polished2 > best_r:
                        best_r = r_polished2
                        best_centers = polished2.copy()
            except:
                pass
    
    return best_centers
