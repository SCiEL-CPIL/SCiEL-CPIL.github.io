"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the packing radius for given centers."""
    n = len(points)
    # Distance to boundary
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j]) / 2.0
            if d < r:
                r = d
    return r

def _score_vec(points):
    """Vectorized score computation."""
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    # Try to find a good spacing
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        # Generate hex lattice points
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            # Sort by distance from center, take closest n
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        # Fallback: random
        best_points = np.random.randn(n, 2) * 0.5
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 20s safety margin
    
    # Start with hexagonal initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # Local optimization using L-BFGS-B
    # We maximize r, equivalent to minimizing -r
    # But r is non-smooth (min of many functions)
    # Use a smooth approximation: minimize sum of -log of constraints or similar
    
    def objective(params):
        pts = params.reshape(-1, 2)
        # Compute all pairwise half-distances and boundary distances
        n = len(pts)
        boundary = 1.0 - np.linalg.norm(pts, axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(n, 1)
        pair_dists = dists[iu] / 2.0
        all_dists = np.concatenate([boundary, pair_dists])
        # Use a smooth min approximation: -logsumexp of -dists scaled
        # Or simply use the actual min but with a smooth surrogate
        # For L-BFGS-B, we need a smooth objective
        # Use: minimize -min(all_dists) is non-smooth
        # Alternative: minimize sum of (max(0, target - d))^2 for some target
        # Better: use the fact that we want to maximize the minimum
        # Use a log-sum-exp based smooth minimum
        # min(x) ≈ -1/b * log(sum(exp(-b*x)))
        b = 50.0  # sharpness parameter
        neg_dists = -all_dists
        max_neg = np.max(neg_dists)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
        smooth_min = -log_sum / b
        return -smooth_min  # minimize negative smooth min
    
    # Try L-BFGS-B optimization
    x0 = points.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() < deadline:
            cand = res.x.reshape(-1, 2)
            # Ensure validity
            if np.all(np.isfinite(cand)):
                cand_score = _score_vec(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
    except Exception:
        pass
    
    # Multi-start local optimization
    n_restarts = 0
    while time.time() < deadline - 10.0:
        # Generate a new random starting point near the current best
        if n_restarts == 0:
            # First restart: use the current best
            x0 = best_points.flatten()
        else:
            # Perturb the best point
            perturbation_scale = 0.02 / (1.0 + 0.1 * n_restarts)
            x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if time.time() >= deadline - 10.0:
                break
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score_vec(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
