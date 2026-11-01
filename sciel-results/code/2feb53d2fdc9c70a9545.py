"""Circle packing: 50 equal circles in unit disk.
Strategy: Smooth surrogate gradient ascent + local Nelder-Mead polish."""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)

def _softmin(x, t):
    """Soft minimum: -t * log(sum(exp(-x_i/t)))."""
    min_val = np.min(x)
    return -t * np.log(np.sum(np.exp(-(x - min_val) / t))) + min_val

def _objective_and_grad(x, t):
    """
    Objective: -softmin(min_pairwise/2, min_boundary)
    Gradient: -d(softmin)/dx
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary term: 1 - |p_i|
    r = np.linalg.norm(pts, axis=1)
    r_safe = np.maximum(r, 1e-12)
    boundary_dists = 1.0 - r
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize min( min_pairwise/2, min_boundary )
    # Let y be the vector of all relevant distances:
    # y = [d_ij/2 for i<j, 1-|p_i| for i]
    # We want softmin(y).
    
    # Vectorized construction of y is memory heavy for N=50?
    # N=50 -> 1225 pairs + 50 boundary = 1275 values.
    # This is small enough to keep in memory.
    
    # Flatten upper triangle of dists/2
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Softmin value
    min_val = np.min(all_dists)
    exp_terms = np.exp(-(all_dists - min_val) / t)
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = -t * log_sum + min_val
    
    # Gradient of softmin w.r.t x
    # d(softmin)/dx_k = - (1/t) * sum_j (exp(-y_j/t) / sum(exp(-y_j/t))) * dy_j/dx_k
    # Let w_j = exp(-(y_j - min_val)/t) / sum(exp(...))  (softmax weights)
    w = exp_terms / np.sum(exp_terms)
    
    # Gradient of y_j w.r.t x_k
    # For pair (i, j): y = |p_i - p_j|/2
    # dy/dp_i = (p_i - p_j) / |p_i - p_j|
    # dy/dp_j = (p_j - p_i) / |p_i - p_j|
    
    # For boundary i: y = 1 - |p_i|
    # dy/dp_i = -p_i / |p_i|
    
    grad = np.zeros_like(x)
    
    # Pairwise gradients
    # diff[i, j] = p_i - p_j
    # dists[i, j] = |p_i - p_j|
    # We need to sum w_k * (p_i - p_j)/|p_i - p_j| for all pairs (i, j)
    
    # Vectorized approach:
    # For each point i, sum over all j != i:
    # w_{ij} * (p_i - p_j) / d_{ij}
    
    # Let's compute contributions per point
    # W matrix: w[i, j] = w_{ij} for i<j, 0 otherwise? No, w is a flat vector.
    # Let's map flat index back to (i, j).
    
    # This is slightly complex to vectorize fully without loops over N or pairs.
    # N=50 is small. A loop over pairs (1225) is fast.
    
    # Alternative: Use the fact that we can compute gradient via automatic differentiation? No, scipy doesn't do that here.
    # Let's do a semi-vectorized loop over points.
    
    # For each point i:
    # grad[i] += sum_{j != i} w_{ij} * (p_i - p_j)/d_{ij}
    # grad[i] += w_{boundary_i} * (-p_i/|p_i|)
    
    # We have w_flat.
    # w_flat[0:1225] correspond to pairs in triu order.
    # w_flat[1225:1275] correspond to boundary terms.
    
    # Let's just loop over points and sum contributions.
    # Precompute unit vectors for pairs?
    # unit_diff[i, j] = (p_i - p_j) / d_{ij}
    
    unit_diff = diff / np.maximum(dists, 1e-12)[:, :, np.newaxis]
    np.fill_diagonal(unit_diff, 0.0)
    
    # We need to apply weights w to the pairs.
    # w_pair is shape (1225,).
    # We need to sum w_pair[k] * unit_diff[i, j] for all (i, j) corresponding to k.
    
    # This is a weighted sum of rows/cols.
    # Let's construct a weight matrix W_pair of shape (N, N) where W_pair[i, j] = w_{ij} if i<j, w_{ji} if i>j, 0 if i=j.
    W_pair = np.zeros((n, n))
    W_pair[i_idx, j_idx] = w[:len(i_idx)]
    W_pair[j_idx, i_idx] = w[:len(i_idx)]
    
    # grad_pair[i] = sum_j W_pair[i, j] * unit_diff[i, j]
    grad_pair = np.sum(W_pair[:, :, np.newaxis] * unit_diff, axis=1)
    
    # Boundary gradient
    # w_boundary is shape (N,)
    w_boundary = w[len(i_idx):]
    boundary_unit = -pts / r_safe[:, np.newaxis]
    grad_boundary = w_boundary[:, np.newaxis] * boundary_unit
    
    total_grad_pts = grad_pair + grad_boundary
    
    # Objective is -softmin, so gradient of objective is -gradient of softmin
    obj = -softmin_val
    grad = -total_grad_pts.flatten()
    
    return obj, grad

def _gradient_ascent(points, t=0.05, max_iter=200, deadline=None):
    """
    Optimize using L-BFGS-B on the negative soft-min objective.
    """
    x0 = points.flatten()
    
    def obj_and_grad(x):
        return _objective_and_grad(x, t)
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True, # We provide gradient in function
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-8, 'maxfun': 5000}
        )
        return res.x.reshape(-1, 2)
    except Exception as e:
        # print(e)
        return points

def _nelder_mead_polish(points, max_iter=1000, deadline=None):
    """
    Polish using Nelder-Mead on the raw min-score objective.
    """
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalty for going outside
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return 10.0 + np.sum(dists[dists > 1.0] - 1.0)
        return -_compute_score(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        new_points = res.x.reshape(-1, 2)
        # Ensure validity
        if np.all(np.linalg.norm(new_points, axis=1) < 1.0):
            return new_points
    except Exception:
        pass
    return points

def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _hex_init(n, seed=0):
    """Hexagonal lattice init."""
    rng = np.random.RandomState(seed)
    best_spacing = None
    best_points = None
    
    # Try different spacings
    for spacing in np.linspace(0.12, 0.25, 15):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= n:
            order = np.argsort(radius[mask], kind="stable")
            selected = points[order[:n]]
            score = _compute_score(selected)
            if best_points is None or score > _compute_score(best_points):
                best_points = selected.copy()
                best_spacing = spacing
    
    if best_points is not None:
        return best_points
    
    return _random_init(n, seed)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Strategy:
    # 1. Generate multiple diverse initializations (Hex + Random)
    # 2. For each, run Gradient Ascent (smooth) to get close to a local optimum
    # 3. Polish with Nelder-Mead (non-smooth) for final precision
    
    inits = []
    # 2 Hex inits
    inits.append(_hex_init(N, seed=0))
    inits.append(_hex_init(N, seed=1))
    # 3 Random inits
    for s in range(5):
        inits.append(_random_init(N, seed=s))
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Step 1: Gradient Ascent with decreasing temperature
        # Start with larger t to escape local minima, decrease t for precision
        current_pts = init_pts.copy()
        
        # Annealing schedule for t
        ts = [0.1, 0.05, 0.01, 0.001]
        
        for t in ts:
            if time.time() > deadline - 5.0:
                break
            current_pts = _gradient_ascent(current_pts, t=t, max_iter=100, deadline=deadline)
            
        # Step 2: Nelder-Mead Polish
        if time.time() < deadline - 2.0:
            polished_pts = _nelder_mead_polish(current_pts, max_iter=500, deadline=deadline)
            score = _compute_score(polished_pts)
            if score > best_score:
                best_score = score
                best_points = polished_pts.copy()
        
        # Also check the unpolished point just in case
        score_unpolished = _compute_score(current_pts)
        if score_unpolished > best_score:
            best_score = score_unpolished
            best_points = current_pts.copy()
            
    # Fallback
    if best_points is None:
        best_points = _hex_init(N, seed=0)
        
    return best_points

# EVOLVE-BLOCK-END
