"""Packing 50 equal circles in a unit circle using gradient-based optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(pts):
    """Compute the actual packing score: min of half pairwise distances and boundary distances."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _soft_min_obj(pts, beta, sym_penalty=0.0):
    """
    Soft-minimum surrogate for the packing objective.
    Uses log-sum-exp to approximate the min of:
      - half pairwise distances
      - boundary distances (1 - |c_i|)
    Minimizing -soft_min is equivalent to maximizing the soft_min.
    """
    n = len(pts)
    # Pairwise distances (only i < j to avoid double counting)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pts[idx_i] - pts[idx_j]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Combine all "constraints" into one array
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft minimum via log-sum-exp: soft_min = -log(sum(exp(-x/beta)))/beta
    # For numerical stability, shift by min
    x = all_dists
    x_min = np.min(x)
    log_sum = np.log(np.sum(np.exp(-(x - x_min) / beta)))
    soft_min_val = x_min - beta * log_sum
    
    # Symmetry-breaking penalty: penalize points that are too close to symmetric positions
    # This helps escape symmetric local minima
    if sym_penalty > 0.0:
        # Penalize points near the center and near symmetric axes
        center_dist = np.sqrt(np.sum(pts ** 2, axis=1))
        # Small penalty for being exactly at center
        sym_pen = sym_penalty * np.sum(np.exp(-center_dist ** 2 / 0.01))
        soft_min_val -= sym_pen
    
    return -soft_min_val  # We minimize the negative

def _soft_min_grad(pts, beta, sym_penalty=0.0):
    """Gradient of the soft-minimum objective with respect to point positions."""
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Pairwise distances (i < j)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pts[idx_i] - pts[idx_j]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min weights: w_k = exp(-x_k/beta) / sum(exp(-x_j/beta))
    x = all_dists
    x_min = np.min(x)
    exp_vals = np.exp(-(x - x_min) / beta)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min w.r.t. x: d(soft_min)/dx_k = -w_k
    # Then chain rule through x_k to pts
    
    # Pairwise part: x_k = |p_i - p_j| / 2 for pair (i,j)
    # d(x_k)/d(p_i) = -(p_i - p_j) / (2 * |p_i - p_j|) = -diff / (2 * 2*|diff|) ... 
    # Actually: x_k = 0.5 * ||p_i - p_j||, so d(x_k)/d(p_i) = 0.5 * (p_i - p_j)/||p_i - p_j||
    # and d(x_k)/d(p_j) = 0.5 * (p_j - p_i)/||p_i - p_j|| = -d(x_k)/d(p_i)
    
    n_pairs = len(pair_dists)
    # For each pair, compute gradient contribution
    for k in range(n_pairs):
        i, j = idx_i[k], idx_j[k]
        d = diff[k]  # p_i - p_j
        norm = np.sqrt(np.sum(d ** 2))
        if norm < 1e-12:
            continue
        # d(x_k)/d(p_i) = 0.5 * d / norm
        # d(x_k)/d(p_j) = -0.5 * d / norm
        # d(soft_min)/d(p_i) = -w_k * 0.5 * d / norm
        # d(soft_min)/d(p_j) = -w_k * (-0.5 * d / norm) = w_k * 0.5 * d / norm
        w = weights[k]
        factor = w * 0.5 / norm
        grad[i] -= factor * d  # because we're minimizing -soft_min, so sign flips
        grad[j] += factor * d
    
    # Boundary part: x_k = 1 - |p_i| for point i
    # d(x_k)/d(p_i) = -p_i / |p_i|
    for k in range(n):
        i = k
        r = radii[i]
        if r < 1e-12:
            continue
        w = weights[n_pairs + k]
        # d(soft_min)/d(p_i) = -w * (-p_i / r) = w * p_i / r
        # Minimizing -soft_min: gradient = -d(soft_min)/d(p_i) = -w * p_i / r
        grad[i] -= w * pts[i] / r
    
    # Symmetry-breaking gradient
    if sym_penalty > 0.0:
        center_dist = np.sqrt(np.sum(pts ** 2, axis=1))
        # d/dp of sum(exp(-|p|^2/0.01)) = sum(-2*p/0.01 * exp(-|p|^2/0.01))
        # Objective has -sym_pen, so gradient += sym_penalty * 2*p/0.01 * exp(...)
        exp_terms = np.exp(-center_dist ** 2 / 0.01)
        grad += sym_penalty * (2.0 / 0.01) * pts * exp_terms[:, np.newaxis]
    
    return grad

def _hexagonal_init(n_points, n_rings=8):
    """Generate initial points on a hexagonal lattice pattern within the unit circle."""
    points = []
    # Add center
    points.append([0.0, 0.0])
    
    # Add rings of points
    ring_radius = 0.15
    for ring in range(1, n_rings + 1):
        r = ring * ring_radius
        if r > 0.95:
            break
        # Number of points in this ring
        n_in_ring = max(6, int(2 * np.pi * r / (2 * ring_radius * 0.9)))
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            if x * x + y * y <= 0.95 ** 2:
                points.append([x, y])
    
    # If we have too many, subsample; if too few, add more
    if len(points) > n_points:
        # Take first n_points
        points = points[:n_points]
    elif len(points) < n_points:
        # Add random points near the boundary
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.7, 0.95)
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(points[:n_points], dtype=np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Initial guess: hexagonal pattern
    np.random.seed(42)
    pts = _hexagonal_init(N)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Adaptive beta schedule: start with small beta (sharp min), increase over time
    # Actually, for optimization, we want to start with larger beta (smoother) and decrease
    # to converge to the true min. Let's use a schedule.
    
    beta_start = 0.05
    beta_end = 0.005
    
    # Phase 1: Coarse optimization with larger beta
    n_coarse_iter = 3
    
    for phase in range(n_coarse_iter):
        if time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (phase / max(1, n_coarse_iter - 1))
        
        # Symmetry-breaking: stronger in early phases
        sym_pen = 0.01 * (1.0 - phase / n_coarse_iter)
        
        def obj_func(x):
            return _soft_min_obj(x.reshape(-1, 2), beta, sym_pen)
        
        def grad_func(x):
            return _soft_min_grad(x.reshape(-1, 2), beta, sym_pen)
        
        x0 = best_pts.flatten()
        
        try:
            result = minimize(
                obj_func,
                x0,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            new_pts = result.x.reshape(-1, 2)
            
            # Validate
            if np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Fine optimization with smaller beta, no symmetry penalty
    n_fine_iter = 5
    for phase in range(n_fine_iter):
        if time.time() > deadline:
            break
        
        beta = beta_end * (0.5 ** phase)  # Decrease beta further
        beta = max(beta, 0.001)
        
        def obj_func(x):
            return _soft_min_obj(x.reshape(-1, 2), beta, 0.0)
        
        def grad_func(x):
            return _soft_min_grad(x.reshape(-1, 2), beta, 0.0)
        
        x0 = best_pts.flatten()
        
        try:
            result = minimize(
                obj_func,
                x0,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
            )
            new_pts = result.x.reshape(-1, 2)
            
            if np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement - try to improve by direct pair adjustments
    if time.time() < deadline - 10:
        # Simple local search: for each pair, try to increase their distance
        for iteration in range(20):
            if time.time() > deadline:
                break
            
            improved = False
            # Compute all pair distances
            n = len(best_pts)
            for i in range(n):
                for j in range(i + 1, n):
                    d = best_pts[i] - best_pts[j]
                    dist = np.sqrt(np.sum(d ** 2))
                    if dist < 1e-10:
                        continue
                    
                    # Try small perturbations to increase this distance
                    # Move i away from j and j away from i
                    dir_i = d / dist
                    dir_j = -d / dist
                    
                    # Small step
                    step = 0.001
                    new_i = best_pts[i] + step * dir_i
                    new_j = best_pts[j] + step * dir_j
                    
                    # Check if this improves the overall score
                    test_pts = best_pts.copy()
                    test_pts[i] = new_i
                    test_pts[j] = new_j
                    
                    # Quick check: just check if new_i and new_j are still in circle
                    if np.sum(new_i ** 2) <= 1.0 and np.sum(new_j ** 2) <= 1.0:
                        test_score = _compute_score(test_pts)
                        if test_score > best_score + 1e-12:
                            best_pts = test_pts
                            best_score = test_score
                            improved = True
            
            if not improved:
                break
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        # Fallback to initial
        best_pts = _hexagonal_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
