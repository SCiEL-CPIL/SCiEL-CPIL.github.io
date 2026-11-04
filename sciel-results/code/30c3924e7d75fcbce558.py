"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Polar coordinate parameterization with angular sorting to enforce convexity,
optimized via smooth log-sum-exp surrogate using L-BFGS-B gradient solver.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

def _compute_score_and_worst(pts):
    """
    Calculate the score and identify the worst triangle.
    Returns (score, worst_indices) where worst_indices is a tuple (i, j, k) or None.
    """
    try:
        # Check for degeneracy
        if np.allclose(pts, pts[0]):
            return 0.0, None
            
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0, None
        
    if hull_area < 1e-10:
        return 0.0, None

    n = len(pts)
    min_area = float('inf')
    worst_idx = None
    
    # Vectorized computation for speed
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    worst_idx = (i[min_idx], j[min_idx], k[min_idx])
    
    score = min_area / hull_area
    return score, worst_idx

def _compute_score(pts):
    """
    Wrapper for score only.
    """
    score, _ = _compute_score_and_worst(pts)
    return score

def _decode_params(x):
    """
    Decode parameter vector x into 2D points.
    x[0:N] are radii, x[N:2*N] are angles.
    Sorts points by angle to enforce convex order (radial sort).
    """
    radii = x[:N]
    angles = x[N:2*N]
    
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_radii = radii[sort_idx]
    
    pts = np.column_stack([
        sorted_radii * np.cos(sorted_angles),
        sorted_radii * np.sin(sorted_angles)
    ])
    return pts

def _objective(x):
    """
    Objective function: Negative smooth minimum triangle area normalized by hull area.
    Uses log-sum-exp for smoothness.
    """
    pts = _decode_params(x)
    
    # Compute all triangle areas
    n = N
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas == 0):
        return 1e10 # Penalty for degenerate triangles

    # Compute hull area
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10

    # Smooth min using log-sum-exp: -log(sum(exp(-areas/tau))) * tau
    # We want to maximize min(areas) / hull_area
    # Equivalent to minimizing -min(areas) / hull_area
    # Let's minimize: - (smooth_min(areas) / hull_area)
    
    tau = 0.05 # Temperature for smoothing
    # To avoid overflow, shift by max
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    smooth_min = max_area - tau * log_sum_exp
    
    score = smooth_min / hull_area
    return -score

def _construct_initial_params(seed):
    """
    Generate initial parameters (radii, angles) for optimization.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a regular polygon-like structure
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
    radii = np.ones(N) + rng.normal(0, 0.1, N)
    radii = np.clip(radii, 0.1, 10)
    
    x = np.concatenate([radii, angles])
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Multi-start with L-BFGS-B
    num_starts = 20
    time_per_start = (deadline - start_time) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        x0 = _construct_initial_params(seed=1000 + i)
        
        try:
            # Use L-BFGS-B for gradient-based optimization
            # No explicit bounds needed for angles (periodic), but radii should be positive
            # We can clip radii in decode, but for optimizer, let's set loose bounds
            # Actually, L-BFGS-B works best with bounds.
            # Radii: [0.01, 100]
            # Angles: [0, 2*pi] (though periodic, bounds help convergence in local region)
            
            bounds = [(0.01, 100.0)] * N + [(0.0, 2 * np.pi)] * N
            
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                pts_opt = _decode_params(x_opt)
                score_opt, _ = _compute_score_and_worst(pts_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
                    
        except Exception:
            pass

    if best_pts is None:
        # Fallback to Halton
        pts = _halton_sequence(N, 2)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-10:
                pts /= np.sqrt(area)
        except Exception:
            pass
        best_pts = pts

    return best_pts

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1]^2 for a given base."""
    seq = np.zeros((n, 2))
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        for d in range(1, 3):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += rem * f
            seq[i-1, d-1] = r
    return seq

# EVOLVE-BLOCK-END
