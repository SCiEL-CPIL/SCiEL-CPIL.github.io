import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Theoretical max radius is roughly 1/sqrt(N) * sqrt(3/2) for hex packing
    # r_max ~ 0.053 for N=300
    # Spacing s = 2*r
    # Area per point ~ s^2 * sqrt(3)/2
    # Total area ~ N * s^2 * sqrt(3)/2 <= pi * 1^2
    # s <= sqrt(2*pi / (N*sqrt(3)))
    
    base_spacing = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0))) * 0.95
    
    for attempt in range(20):
        # Vary spacing slightly and rotation
        spacing = base_spacing * (1.0 - 0.01 * attempt + 0.005 * rng.random())
        
        k = int(1.0 / spacing) + 5
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Apply random rotation
                angle = rng.uniform(0, 2 * np.pi)
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                r = np.hypot(x_rot, y_rot)
                # Ensure points are well inside to allow for optimization
                if r <= 1.0 - spacing:
                    points.append((x_rot, y_rot))
        
        if len(points) < n_points:
            continue
            
        pts = np.array(points)
        # Select n_points points closest to the center? Or random?
        # Closest to center is usually better for symmetric packings
        dists_center = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists_center, kind="stable")
        selected = pts[order[:n_points]]
        
        # Calculate initial radius
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        # Fallback: random points
        best_pts = rng.uniform(-0.9, 0.9, (n_points, 2))
        best_r = 0.0
        
    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _soft_min(values, T):
    """Stable soft minimum."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum

def _objective_and_grad(x, T):
    """
    Objective: minimize -soft_min(constraints)
    Constraints: pairwise distances / 2 and boundary distances.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary constraints
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise constraints
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    pair_half = d / 2.0
    iu = np.triu_indices(n, k=1)
    pair_vals = pair_half[iu]
    
    all_vals = np.concatenate([pair_vals, boundary])
    
    obj = -_soft_min(all_vals, T)
    
    # Gradient calculation
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        # Hard min case
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights
        
    grad_pts = np.zeros_like(pts)
    
    # Boundary gradient
    # d(1 - ||p_i||) / d p_i = - p_i / ||p_i||
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis]
    grad_pts += grad_boundary * grad_all[len(pair_vals):][:, np.newaxis]
    
    # Pairwise gradient
    # d(||p_i - p_j|| / 2) / d p_i = (p_i - p_j) / (2 * ||p_i - p_j||)
    # We have weights for each pair (i, j)
    # grad_pts[i] += sum_j w_ij * (p_i - p_j) / (2 * d_ij)
    # grad_pts[j] += sum_i w_ij * (p_j - p_i) / (2 * d_ij)
    
    # Vectorized computation for pairwise gradients
    # Create a matrix of weights for upper triangle
    W = np.zeros((n, n))
    W[iu] = grad_all[:len(pair_vals)]
    W[iu[1], iu[0]] = grad_all[:len(pair_vals)]
    
    # Compute unit vectors
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero
    unit_vecs = diff / dists[:, :, np.newaxis]
    
    # Gradient contribution from pairs
    # Note: The derivative of ||p_i - p_j||/2 wrt p_i is (p_i - p_j)/(2*||p_i - p_j||)
    # So we multiply unit_vecs by 0.5 * W
    weighted_unit = 0.5 * W[:, :, np.newaxis] * unit_vecs
    grad_pts += np.sum(weighted_unit, axis=1)
    
    return obj, grad_pts.ravel()

def _local_refine(pts, T=1e-4, max_iter=1000, deadline=None):
    """Refine the solution using L-BFGS-B on the soft-min objective."""
    x0 = pts.ravel()
    
    def obj(x):
        return _objective_and_grad(x, T)[0]
        
    def jac(x):
        return _objective_and_grad(x, T)[1]
        
    try:
        result = minimize(
            obj,
            x0,
            jac=jac,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'maxfun': 1000
            }
        )
        new_pts = result.x.reshape(-1, 2)
        # Check validity
        if np.all(np.isfinite(new_pts)) and np.all(np.linalg.norm(new_pts, axis=1) < 1.0):
            return new_pts
    except Exception:
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = _hex_init(N, seed=42)
    best_r = _compute_radius(best_pts)
    
    # Annealing schedule for T
    # Start with a relatively high T to allow large moves, then decrease
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for T in temperatures:
        if time.time() > deadline - 20.0:
            break
            
        # Use a subset of points or all points? All points is better for global structure.
        # But L-BFGS-B with 600 variables is okay.
        
        # Run optimization
        refined_pts = _local_refine(best_pts, T=T, max_iter=200, deadline=deadline)
        
        r = _compute_radius(refined_pts)
        if r > best_r:
            best_r = r
            best_pts = refined_pts.copy()
            
    # Final polish with very low T and more iterations
    if time.time() < deadline - 5.0:
        refined_pts = _local_refine(best_pts, T=1e-5, max_iter=500, deadline=deadline)
        r = _compute_radius(refined_pts)
        if r > best_r:
            best_r = r
            best_pts = refined_pts.copy()
            
    # Final validity check
    norms = np.linalg.norm(best_pts, axis=1)
    if not np.all(norms < 1.0) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=42)
        
    return best_pts
