# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 600.0  # seconds
_N = 20


def _vertex_score(d):
    """Compute the per-vertex score given sorted distances d (length 19)."""
    if d[0] < 0.05:
        return 0.0
    diffs = d[1:] - d[:-1]
    k = int(np.argmin(diffs))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = np.array([0, 1, 2, 3])
    elif k == 17:
        idx = np.array([15, 16, 17, 18])
    else:
        idx = np.array([k - 1, k, k + 1, k + 2])
    four = d[idx]
    raw = np.mean(np.maximum(four / D, D / four))
    return min(1.0, 1.0 / raw)


def _score_polygon(pts):
    """
    Score a (20,2) array of vertices. Returns (score, is_valid).
    """
    if pts.shape != (20, 2):
        return 0.0, False
    if not np.all(np.isfinite(pts)):
        return 0.0, False

    # Normalize
    centroid = pts.mean(axis=0)
    p = pts - centroid
    dists_to_centroid = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0, False
    p = p / mean_dist

    # Check no repeated points
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return 0.0, False

    # Check strictly convex: all cross products same sign, nonzero
    cross_signs = []
    for i in range(20):
        a = p[i]
        b = p[(i + 1) % 20]
        c = p[(i + 2) % 20]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-12:
            return 0.0, False
        cross_signs.append(np.sign(cross))
    if len(set(cross_signs)) != 1:
        return 0.0, False

    # Compute per-vertex scores
    scores = []
    for i in range(20):
        d = np.sort(np.linalg.norm(p - p[i], axis=1))
        d = d[1:]  # exclude self
        scores.append(_vertex_score(d))

    min_score = min(scores)
    return min_score, True


def _regular_polygon():
    """Fallback: regular 20-gon."""
    i = np.arange(20)
    angles = 2.0 * np.pi * i / 20.0
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _build_polygon_from_angles(angles):
    """
    Build polygon from absolute angles.
    angles: array of 20 values.
    We assume radius is 1.0 for simplicity, as normalization handles scale.
    To ensure convexity and cyclic order, angles must be strictly increasing mod 2pi.
    """
    # Ensure angles are in range and sorted? 
    # The input to the optimizer will be deviations from regular.
    # Here we just map angles to points.
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)


def _objective_angle_devs(devs):
    """
    Objective function for Nelder-Mead.
    devs: array of 20 angular deviations from regular polygon angles.
    """
    base_angles = 2.0 * np.pi * np.arange(20) / 20.0
    angles = base_angles + devs
    
    # Ensure convexity by checking if the polygon formed is strictly convex.
    # A simple check: the sorted angles must wrap around 2pi exactly once without backtracking.
    # Since we start from regular and deviations are small, we assume cyclic order is preserved.
    # However, we must ensure the resulting polygon is strictly convex.
    
    pts = _build_polygon_from_angles(angles)
    score, valid = _score_polygon(pts)
    if not valid:
        return 1.0  # Penalty for invalid
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum vertex equidistance score.
    Strategy: Multi-start Nelder-Mead optimization on angular deviations.
    """
    best_pts = _regular_polygon()
    best_score, _ = _score_polygon(best_pts)
    
    # Time budget for search
    time_budget = _TIME_LIMIT - 30  # Leave 30s safety margin
    start_time = time.time()
    
    # Number of restarts based on time. Each Nelder-Mead run takes ~0.1-1s.
    # We will run a loop of restarts until time is up.
    
    np.random.seed(42)
    
    while time.time() - start_time < time_budget:
        # Initialize deviations with small random noise
        # The optimal solution is likely close to the regular polygon.
        # We use a small range to maintain convexity easily.
        devs = np.random.normal(0, 0.05, 20)
        
        # Optional: Add a slight systematic bias if we know the direction?
        # For now, pure random restarts.
        
        res = minimize(
            _objective_angle_devs, 
            devs, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if res.fun < 0:  # Valid score found
            candidate_devs = res.x
            base_angles = 2.0 * np.pi * np.arange(20) / 20.0
            angles = base_angles + candidate_devs
            candidate_pts = _build_polygon_from_angles(angles)
            
            cand_score, cand_valid = _score_polygon(candidate_pts)
            if cand_valid and cand_score > best_score:
                best_score = cand_score
                best_pts = candidate_pts
                
                # If we found a very good score, we might tighten the search
                # or just continue to see if we can push it higher.
        
        # Check time
        if time.time() - start_time > time_budget:
            break

    return best_pts

# EVOLVE-BLOCK-END
