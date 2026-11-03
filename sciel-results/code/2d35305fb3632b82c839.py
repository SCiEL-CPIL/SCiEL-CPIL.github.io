import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(points):
    """Compute the packing radius for given points."""
    if len(points) != N:
        return 0.0
    # Check finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check inside unit circle
    dists = np.hypot(points[:, 0], points[:, 1])
    if np.any(dists >= 1.0):
        return 0.0
    # Boundary constraint
    boundary = 1.0 - dists
    min_boundary = np.min(boundary)
    # Pairwise distances
    # Use vectorized computation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist_mat, np.inf)
    min_pair = np.min(dist_mat) / 2.0
    return min(min_boundary, min_pair)

def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp trick."""
    values = np.asarray(values)
    # For soft min: -logsumexp(-values/beta) * beta
    # But to avoid overflow, shift
    shifted = -values / beta
    max_val = np.max(shifted)
    if np.isinf(max_val):
        return np.min(values)
    exp_shifted = np.exp(shifted - max_val)
    logsumexp = max_val + np.log(np.sum(exp_shifted))
    return -logsumexp * beta

def _objective_and_grad(x_flat, beta=30.0):
    """Objective function and gradient for optimization.
    
    We minimize the negative soft-min of the constraints.
    """
    points = x_flat.reshape(N, 2)
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-30)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Boundary distances
    dists = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - dists
    
    # Combine all constraints: pairwise/2 and boundary
    # We want soft-min of all these values
    pair_vals = dist_mat / 2.0
    all_vals = np.concatenate([pair_vals.ravel(), boundary])
    
    # Soft min
    soft_min_val = _soft_min(all_vals, beta=beta)
    
    # We want to maximize the min, so minimize -soft_min
    obj = -soft_min_val
    
    # Gradient computation (numerical would be too slow, use analytical approximation)
    # For L-BFGS, we can use numerical gradient or just use method='L-BFGS-B' without explicit grad
    return obj

def _hex_initialization(n_points):
    """Generate hexagonal lattice points inside unit circle."""
    # Find appropriate spacing
    best_spacing = None
    for trial in range(50):
        spacing = 0.15 - trial * 0.003
        if spacing <= 0:
            break
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append([x, y])
        if len(rows) >= n_points:
            points = np.array(rows)
            # Sort by distance from center, take closest n_points
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n_points]
            return points[idx]
    # Fallback: random
    return np.random.uniform(-0.9, 0.9, size=(n_points, 2))

def _optimize_from_start(points0, time_limit=120.0, beta_start=50.0, beta_end=10.0, max_iter=200):
    """Run optimization from given starting points."""
    start_time = time.time()
    x0 = points0.ravel()
    
    # Animate beta from high (smooth) to low (sharper)
    n_anneal = 5
    for anneal_step in range(n_anneal):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:  # Safety margin
            break
        
        beta = beta_start * (beta_end / beta_start) ** (anneal_step / max(n_anneal - 1, 1))
        
        # Use L-BFGS-B with numerical gradient
        result = minimize(
            _objective_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': max_iter // n_anneal, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50},
        )
        
        x0 = result.x
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
    
    return x0.reshape(N, 2)

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 10.0
    
    # Get initial hexagonal packing
    points0 = _hex_initialization(N)
    best_points = points0.copy()
    best_score = _compute_score(best_points)
    
    # Try multiple optimization runs with different initializations
    n_restarts = 3
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - safety_margin:
            break
        
        if restart == 0:
            start_points = points0.copy()
        else:
            # Add small random perturbation to escape local minima
            start_points = points0 + np.random.normal(0, 0.01, size=(N, 2))
            # Ensure inside circle
            dists = np.hypot(start_points[:, 0], start_points[:, 1])
            mask = dists > 0.95
            start_points[mask] *= 0.95 / dists[mask]
        
        optimized = _optimize_from_start(start_points, time_limit=time_limit - elapsed - 5)
        score = _compute_score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
        
        elapsed = time.time() - start_time
        if elapsed > time_limit - safety_margin:
            break
    
    # Final polish: one more optimization with lower beta
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20:
        polished = _optimize_from_start(best_points, time_limit=time_limit - elapsed - 5, 
                                         beta_start=20.0, beta_end=5.0, max_iter=100)
        score = _compute_score(polished)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    return best_points
