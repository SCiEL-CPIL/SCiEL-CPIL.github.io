import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers."""
    n = len(pts)
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Get pairwise distances excluding diagonal
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    min_pair_dist = np.min(pair_dists)
    
    # Radius is min of half the min pair distance and min boundary distance
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_radius(pts, beta=50.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    We want to maximize r = min(min_pair_dist/2, min_boundary)
    So we minimize -r.
    
    Using soft-min: soft_min(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    For multiple values, we can use log-sum-exp trick.
    
    Actually, let's define it differently. We want to maximize the minimum of:
    - All pairwise distances / 2
    - All boundary distances
    
    Let's use: f(x) = -soft_max(-values) where values are the distances we want to maximize.
    soft_max(v) = 1/beta * log(sum(exp(beta*v_i)))
    So -soft_max(-v) = 1/beta * log(sum(exp(-beta*v_i)))
    """
    n = len(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise distances / 2
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all values we want to maximize
    all_vals = np.concatenate([pair_dists, boundary_dists])
    
    # Soft max of negative values, negated
    # soft_max(v) = 1/beta * log(sum(exp(beta*v)))
    # We want -soft_max(-all_vals) = 1/beta * log(sum(exp(-beta*all_vals)))
    
    # Use log-sum-exp for numerical stability
    max_val = np.max(all_vals)
    log_sum = np.log(np.sum(np.exp(-beta * (all_vals - max_val)))) + (-beta * max_val)
    
    soft_min = -log_sum / beta
    
    return -soft_min  # We're minimizing, so return negative of what we want to maximize

def _soft_min_radius_grad(pts, beta=50.0):
    """Gradient of the soft-min radius objective."""
    n = len(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise distances / 2
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all values
    all_vals = np.concatenate([pair_dists, boundary_dists])
    
    # Soft max gradient
    max_val = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - max_val))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # These are softmax weights
    
    # Gradient of -soft_min = -1/beta * d/dx log(sum(exp(-beta*vals)))
    # = -1/beta * sum_i (-beta * w_i * d(vals_i)/dx)
    # = sum_i w_i * d(vals_i)/dx
    
    grad = np.zeros_like(pts)
    
    # Boundary part
    for i in range(n):
        w = weights[n*(n-1)//2 + i]  # Index in all_vals
        if norms[i] > 1e-10:
            grad[i] += w * (pts[i] / norms[i])  # d(1 - |x_i|)/dx_i = -x_i/|x_i|, but we have +boundary_dist, so +x_i/|x_i|
        # Actually: d(boundary_dist_i)/d(pts[i]) = -pts[i]/norms[i]
        # But we're computing gradient of the objective which is -soft_min
        # Let me redo this more carefully.
    
    # Let me recompute the gradient properly
    # Objective: f(x) = -soft_min_radius = 1/beta * log(sum(exp(-beta*vals)))
    # df/dx = 1/beta * sum_i exp(-beta*vals_i) * (-beta * d(vals_i)/dx) / sum_j exp(-beta*vals_j)
    # = -sum_i w_i * d(vals_i)/dx where w_i = exp(-beta*vals_i) / sum_j exp(-beta*vals_j)
    
    grad = np.zeros_like(pts)
    
    # Pairwise part
    pair_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            w = weights[pair_idx]
            pair_idx += 1
            # d(pair_dists[i,j])/d(pts[i]) = (pts[i] - pts[j]) / (2 * dists[i,j]) * 2? No
            # pair_dists = dists/2, so d(pair_dists)/d(pts[i]) = (pts[i]-pts[j])/dists[i,j] * 1/2 * 2? 
            # Actually: d(|p_i - p_j|/2)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
            if dists[i, j] > 1e-10:
                grad[i] += w * (pts[i] - pts[j]) / (2 * dists[i, j])
                grad[j] -= w * (pts[i] - pts[j]) / (2 * dists[i, j])
    
    # Boundary part
    for i in range(n):
        w = weights[n*(n-1)//2 + i]
        if norms[i] > 1e-10:
            # d(1 - |p_i|)/d(p_i) = -p_i / |p_i|
            grad[i] += w * (-pts[i] / norms[i])
    
    return grad

def _hex_init(n_points, radius=1.0):
    """Generate initial points from a hexagonal lattice clipped to a circle."""
    points = []
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    k_max = int(2.0 / s) + 2
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = i * s + 0.5 * s * (j % 2)
            y = j * s * np.sqrt(3) / 2
            points.append([x, y])
            
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < radius - s/2
    points = points[mask]
    
    if len(points) < n_points:
        k_max *= 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = i * s + 0.5 * s * (j % 2)
                y = j * s * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        distances = np.linalg.norm(points, axis=1)
        mask = distances < radius
        points = points[mask]
        
    distances = np.linalg.norm(points, axis=1)
    indices = np.argsort(distances)
    points = points[indices[:n_points]]
    
    return points

def _ring_init(n_points, radius=1.0):
    """Generate initial points on concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    # Remaining points on rings
    remaining = n_points - 1
    ring_radius = 0.2
    while remaining > 0 and ring_radius < radius:
        # How many points fit on this ring?
        circumference = 2 * np.pi * ring_radius
        # Approximate spacing
        spacing = 0.126 * 2  # Approximate diameter
        n_on_ring = max(1, int(circumference / spacing))
        
        # Take min of what fits and what's remaining
        n_take = min(n_on_ring, remaining)
        
        for k in range(n_take):
            angle = 2 * np.pi * k / n_on_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            points.append([x, y])
        
        remaining -= n_take
        ring_radius += spacing * 0.8
    
    # If we still need more points, add them on the outer ring
    if remaining > 0:
        outer_r = radius - 0.126
        for k in range(remaining):
            angle = 2 * np.pi * k / remaining
            x = outer_r * np.cos(angle)
            y = outer_r * np.sin(angle)
            points.append([x, y])
    
    return np.array(points[:n_points])

def _random_init(n_points, radius=1.0, seed=None):
    """Generate random initial points inside the circle."""
    if seed is not None:
        np.random.seed(seed)
    
    points = []
    for _ in range(n_points):
        r = np.sqrt(np.random.rand()) * (radius - 0.13)
        theta = np.random.rand() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    
    return np.array(points)

def _optimize_from_init(x0, deadline, beta_start=20.0, beta_end=100.0):
    """Optimize from a given initial point using L-BFGS-B with smooth objective."""
    x0_flat = x0.flatten()
    
    best_x = x0_flat
    best_score = _compute_radius(x0)
    
    # Multi-stage optimization with increasing beta
    n_stages = 3
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        def obj(x):
            pts = x.reshape(N, 2)
            return _soft_min_radius(pts, beta=beta)
        
        def grad(x):
            pts = x.reshape(N, 2)
            return _soft_min_radius_grad(pts, beta=beta).flatten()
        
        try:
            res = minimize(obj, x0_flat, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_score:
                pts = res.x.reshape(N, 2)
                # Check validity
                norms = np.linalg.norm(pts, axis=1)
                if np.all(norms < 1.0):
                    r = _compute_radius(pts)
                    if r > best_score:
                        best_score = r
                        best_x = res.x
        except Exception:
            pass
        
        x0_flat = best_x
    
    return best_x.reshape(N, 2), best_score

def _nelder_mead_polish(x0, deadline, maxiter=500):
    """Final polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0_flat = x0.flatten()
    
    def obj(x):
        pts = x.reshape(N, 2)
        return -_compute_radius(pts)
    
    try:
        res = minimize(obj, x0_flat, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
        
        pts = res.x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        if np.all(norms < 1.0):
            r = _compute_radius(pts)
            if r > 0:
                return pts, r
    except Exception:
        pass
    
    return x0, _compute_radius(x0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    best_x = None
    best_score = 0.0
    
    # Try multiple initialization strategies
    inits = []
    
    # Hex lattice
    try:
        hex_pts = _hex_init(N)
        inits.append(('hex', hex_pts))
    except Exception:
        pass
    
    # Ring
    try:
        ring_pts = _ring_init(N)
        inits.append(('ring', ring_pts))
    except Exception:
        pass
    
    # Random with different seeds
    for seed in [42, 123, 456]:
        try:
            rand_pts = _random_init(N, seed=seed)
            inits.append((f'rand_{seed}', rand_pts))
        except Exception:
            pass
    
    # Optimize from each initialization
    for name, init_pts in inits:
        if time.time() > deadline - 20:
            break
        
        try:
            x_opt, score = _optimize_from_init(init_pts, deadline)
            if score > best_score:
                best_score = score
                best_x = x_opt
        except Exception:
            pass
    
    # Final polish with Nelder-Mead if we have time
    if best_x is not None and time.time() < deadline - 15:
        try:
            x_polished, score_polished = _nelder_mead_polish(best_x, deadline, maxiter=300)
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished
        except Exception:
            pass
    
    # If we never got a valid result, return hex init
    if best_x is None:
        best_x = _hex_init(N)
    
    return best_x
