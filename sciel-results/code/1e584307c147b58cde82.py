"""Packing 300 equal circles in a unit circle using soft-min surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hex_lattice_init(n_points, radius=1.0):
    """Generate initial points on a hexagonal lattice within the unit disk."""
    # Estimate spacing for n_points in a hex lattice
    # Area of hex lattice per point ~ (sqrt(3)/2) * s^2
    # Area of unit disk = pi
    # n_points * (sqrt(3)/2) * s^2 ≈ pi * (1 - s/2)^2
    # Solve approximately
    s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0) / 2.0))
    s = s * 0.95  # slight shrink to stay inside
    
    # Generate hex lattice points
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= radius - s / 2.0:
                points.append((x, y))
    
    points = np.array(points)
    if len(points) < n_points:
        # If not enough, add random points
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius - s / 2.0)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    # Select n_points closest to origin for better initial packing
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind='stable')[:n_points]
    return points[idx]


def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and distance to boundary."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _soft_min_obj(points, beta, target_r):
    """
    Soft-min objective using log-sum-exp.
    Minimizes the negative of the soft-min of (d_ij/2 - target_r) and (1 - |c_i| - target_r).
    As beta -> inf, this approaches the true min.
    """
    n = len(points)
    
    # Pairwise terms: d_ij / 2 - target_r
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0 - target_r
    
    # Boundary terms: 1 - |c_i| - target_r
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_terms = 1.0 - radii - target_r
    
    # Combine all terms
    all_terms = np.concatenate([pair_terms.ravel(), boundary_terms.ravel()])
    # Filter out inf
    all_terms = all_terms[np.isfinite(all_terms)]
    
    if len(all_terms) == 0:
        return 0.0
    
    # Soft min: -log-sum-exp(-beta * x) / beta
    # For numerical stability, use max shift
    shifted = -beta * all_terms
    max_shift = np.max(shifted)
    if max_shift > 700:
        # Avoid overflow
        return np.max(all_terms)
    
    lse = max_shift + np.log(np.sum(np.exp(shifted - max_shift)))
    soft_min = -lse / beta
    
    return -soft_min  # We minimize, so negate


def _soft_min_grad(points, beta, target_r):
    """Gradient of the soft-min objective."""
    n = len(points)
    
    # Pairwise terms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0 - target_r
    
    # Boundary terms
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_terms = 1.0 - radii - target_r
    
    all_terms = np.concatenate([pair_terms.ravel(), boundary_terms.ravel()])
    all_terms = all_terms[np.isfinite(all_terms)]
    
    if len(all_terms) == 0:
        return np.zeros_like(points)
    
    # Soft min weights
    shifted = -beta * all_terms
    max_shift = np.max(shifted)
    if max_shift > 700:
        # Degenerate case
        return np.zeros_like(points)
    
    exp_shifted = np.exp(shifted - max_shift)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of soft_min w.r.t. all_terms is -weights
    # d(soft_min)/d(point) = sum over terms of (-weights) * d(term)/d(point)
    # We want d(-soft_min)/d(point) = sum over terms of weights * d(term)/d(point)
    
    grad = np.zeros_like(points)
    
    # Pairwise gradient
    pair_flat = pair_terms.ravel()
    n_pair = len(pair_flat)
    
    if n_pair > 0:
        pair_weights = weights[:n_pair]
        
        # For each pair (i,j), d(d_ij/2)/d(points[i]) = -(points[i] - points[j]) / (2 * d_ij)
        # d(d_ij/2)/d(points[j]) = (points[i] - points[j]) / (2 * d_ij)
        
        # Vectorized: for each pair, accumulate gradient
        for idx in range(n_pair):
            i, j = np.unravel_index(idx, (n, n))
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            w = pair_weights[idx]
            # d(pair_term)/d(points[i]) = -(points[i] - points[j]) / (2 * d)
            # d(pair_term)/d(points[j]) = (points[i] - points[j]) / (2 * d)
            diff_ij = (points[i] - points[j]) / (2.0 * d)
            grad[i] -= w * diff_ij
            grad[j] += w * diff_ij
    
    # Boundary gradient
    boundary_flat = boundary_terms.ravel()
    n_boundary = len(boundary_flat)
    
    if n_boundary > 0:
        boundary_weights = weights[n_pair:]
        # d(1 - |c_i|)/d(points[i]) = -points[i] / |c_i|
        for i in range(n_boundary):
            r = radii[i]
            if r < 1e-12:
                continue
            w = boundary_weights[i]
            grad[i] -= w * points[i] / r
    
    return grad


def construct():
    """Construct 300 circle centers in the unit disk maximizing the minimum radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial guess: hexagonal lattice
    points = _hex_lattice_init(N)
    
    # Compute initial score
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Multi-stage optimization with increasing beta
    # Start with low beta for smooth landscape, increase for sharper
    betas = [5, 10, 20, 40, 80, 160, 320]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Update target_r to current best score
        target_r = best_score
        
        # Optimize using L-BFGS-B
        def obj(x):
            pts = x.reshape(-1, 2)
            return _soft_min_obj(pts, beta, target_r)
        
        def grad(x):
            pts = x.reshape(-1, 2)
            return _soft_min_grad(pts, beta, target_r).ravel()
        
        x0 = best_points.ravel()
        
        try:
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                     'maxfun': 500})
            
            if result.success or result.fun < obj(x0):
                new_points = result.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
        except Exception:
            pass
    
    # Final polish: try a few more iterations with higher beta
    if time.time() < deadline:
        for beta in [640, 1280]:
            if time.time() > deadline:
                break
            
            target_r = best_score
            
            def obj(x):
                pts = x.reshape(-1, 2)
                return _soft_min_obj(pts, beta, target_r)
            
            def grad(x):
                pts = x.reshape(-1, 2)
                return _soft_min_grad(pts, beta, target_r).ravel()
            
            x0 = best_points.ravel()
            
            try:
                result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12,
                                         'maxfun': 300})
                
                if result.success or result.fun < obj(x0):
                    new_points = result.x.reshape(-1, 2)
                    new_score = _compute_score(new_points)
                    
                    if new_score > best_score:
                        best_points = new_points
                        best_score = new_score
            except Exception:
                pass
    
    # Ensure all points are strictly inside the unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii < 1.0
    if not np.all(mask):
        # Push violating points inward
        for i in range(len(best_points)):
            if radii[i] >= 1.0:
                scale = 0.999 / radii[i] if radii[i] > 0 else 1.0
                best_points[i] *= scale
    
    return best_points
# EVOLVE-BLOCK-END
