import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _get_score(points):
    """
    Computes the score (common radius r) for a given set of points.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if points is None or len(points) != N:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0

    # Boundary constraint: points must be inside unit circle
    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    
    if boundary_r < 0:
        return 0.0

    # Pairwise distances
    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _hexagonal_lattice(spacing):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    """
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _logsumexp_objective_and_grad(x, alpha, lambda_bound):
    """
    Objective: -log(sum_{i<j} exp(-alpha * d_ij)) + lambda * sum(max(0, ||x_i|| - 1)^2)
    Gradient is computed analytically.
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    # Mask for i < j to compute sum once? 
    # The log-sum-exp is over all pairs i != j. 
    # sum_{i<j} exp(-alpha d_ij) = 0.5 * sum_{i!=j} exp(-alpha d_ij)
    # log(0.5 * S) = log(S) - log(2). Constant shift doesn't matter for gradient.
    # So we can just use S = sum_{i!=j} exp(-alpha d_ij).
    
    # exp_terms[i,j] = exp(-alpha * dists[i,j])
    # For i==j, dist=0, exp(0)=1. We should exclude i==j.
    mask = ~np.eye(n, dtype=bool)
    exp_terms = np.zeros_like(dists)
    exp_terms[mask] = np.exp(-alpha * dists[mask])
    
    S = np.sum(exp_terms)
    
    if S == 0:
        # This shouldn't happen with reasonable alpha and distances
        return 0.0, np.zeros_like(x)
    
    # Objective value
    obj_val = -np.log(S)
    
    # Gradient of -log(S)
    # d/dx_i (-log S) = -1/S * dS/dx_i
    # dS/dx_i = sum_{j!=i} d/dx_i (exp(-alpha d_ij))
    # d/dx_i exp(-alpha d_ij) = exp(-alpha d_ij) * (-alpha) * d/dx_i (d_ij)
    # d/dx_i d_ij = (x_i - x_j) / d_ij = diff[i,j] / dists[i,j]
    # So dS/dx_i = -alpha * sum_{j!=i} exp(-alpha d_ij) * (diff[i,j] / dists[i,j])
    
    # factor[i,j] = -alpha * exp_terms[i,j] / dists_safe[i,j]
    # Note: for i==j, exp_terms is 0, so factor is 0.
    factor = np.zeros_like(dists)
    factor[mask] = -alpha * exp_terms[mask] / dists_safe[mask]
    
    grad_logsumexp = - (1.0 / S) * np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary penalty
    norms = np.linalg.norm(points, axis=1)
    # Penalty for going outside unit circle
    violation = np.maximum(0.0, norms - 1.0)
    obj_val += lambda_bound * np.sum(violation ** 2)
    
    grad_boundary = np.zeros_like(points)
    b_mask = norms > 1.0
    if np.any(b_mask):
        norms_safe = np.where(norms[b_mask] < 1e-8, 1e-8, norms[b_mask])
        grad_boundary[b_mask] = 2.0 * lambda_bound * (norms[b_mask] - 1.0) * (points[b_mask] / norms_safe[:, np.newaxis])
    
    total_grad = (grad_logsumexp + grad_boundary).flatten()
    
    return obj_val, total_grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = 0.0
    
    # Parameters for log-sum-exp
    # High alpha makes it focus on the closest pair
    alpha = 50.0 
    lambda_bound = 100.0
    
    # Generate initial candidates
    # Theoretical max r ~ 0.053, so spacing 2r ~ 0.106
    spacings = [0.105, 0.106, 0.107, 0.108, 0.110]
    
    candidates = []
    for sp in spacings:
        pts = _hexagonal_lattice(sp)
        if pts is not None:
            # Add small random noise to break symmetry
            noise = np.random.normal(0, 0.001, pts.shape)
            pts_noisy = pts + noise
            candidates.append(pts_noisy)
            
    # Also try a ring-based initialization
    def _concentric_rings(n_rings):
        if n_rings <= 0:
            return None
        if n_rings == 1:
            angles = 2 * np.pi * np.arange(N) / N
            return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        pts = [[0.0, 0.0]]
        remaining = N - 1
        ring_radii = np.linspace(0.1, 0.95, n_rings - 1)
        weights = ring_radii
        weights /= np.sum(weights)
        counts = np.round(weights * remaining).astype(int)
        diff = remaining - np.sum(counts)
        if diff != 0:
            idx = np.argmax(counts) if diff > 0 else np.argmax(-counts)
            counts[idx] += diff
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r = ring_radii[i]
            angles = 2 * np.pi * np.arange(count) / count
            offset = 0 if i % 2 == 0 else np.pi / count
            x = r * np.cos(angles + offset)
            y = r * np.sin(angles + offset)
            for j in range(count):
                pts.append([x[j], y[j]])
        if len(pts) != N:
            return None
        return np.array(pts)
    
    for nr in [10, 15, 20]:
        pts = _concentric_rings(nr)
        if pts is not None:
            noise = np.random.normal(0, 0.001, pts.shape)
            candidates.append(pts + noise)
    
    if not candidates:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    # Optimize each candidate
    options = {
        'maxiter': 200,
        'ftol': 1e-8,
        'gtol': 1e-5,
        'maxls': 50
    }
    
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        x0 = init_pts.flatten()
        
        try:
            result = minimize(
                _logsumexp_objective_and_grad, 
                x0, 
                args=(alpha, lambda_bound),
                method='L-BFGS-B', 
                jac=True, 
                options=options
            )
            
            opt_points = result.x.reshape(-1, 2)
            score = _get_score(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception as e:
            pass
            
        # If we have time, try to refine the best found so far with a higher alpha
        # to squeeze out more distance
        if time.time() < deadline - 30.0 and best_points is not None:
            # Use the best points as new start, increase alpha to focus on min distance
            alpha_refine = 100.0
            x0_refine = best_points.flatten()
            try:
                result_refine = minimize(
                    _logsumexp_objective_and_grad, 
                    x0_refine, 
                    args=(alpha_refine, lambda_bound),
                    method='L-BFGS-B', 
                    jac=True, 
                    options=options
                )
                opt_points_refine = result_refine.x.reshape(-1, 2)
                score_refine = _get_score(opt_points_refine)
                if score_refine > best_score:
                    best_score = score_refine
                    best_points = opt_points_refine.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback to best initial candidate
        best_points = candidates[0]
        
    return best_points

# EVOLVE-BLOCK-END
