# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex score.
    The scoring rewards vertices where 4 distances to other vertices are nearly equal.
    We use a parameterization by angular deviations from a regular polygon, then optimize.
    """
    n = 20
    deadline = time.time() + 580  # safety margin before 630s kill

    # Helper: compute vertex scores given a polygon (n,2)
    def compute_scores(pts):
        """pts: (n,2) array. Returns array of n vertex scores."""
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        # pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        scores = np.zeros(n)
        for i in range(n):
            d = np.sort(dists[i])  # 19 distances
            if d[0] < 0.05:
                scores[i] = 0.0
                continue
            # find k: first index minimizing d_{k+1} - d_k
            gaps = d[1:] - d[:-1]  # 18 gaps
            k = int(np.argmin(gaps))
            D = (d[k] + d[k + 1]) / 2.0
            if D < 0.1:
                scores[i] = 0.0
                continue
            if k == 0:
                idx = [0, 1, 2, 3]
            elif k == 17:
                idx = [15, 16, 17, 18]
            else:
                idx = [k - 1, k, k + 1, k + 2]
            four_d = d[idx]
            raw = np.mean(np.maximum(four_d / D, D / four_d))
            scores[i] = min(1.0, 1.0 / raw)
        return scores

    def objective(params):
        """params: array of n angular deviations. Returns negative min vertex score."""
        base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles = base_angles + params
        # Ensure angles are in order (they should be if deviations are small)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        # Check convexity: all cross products of consecutive edges should have same sign
        edges = np.roll(pts, -1, axis=0) - pts
        cross = np.cross(edges, np.roll(edges, -1, axis=0))
        if np.any(np.abs(cross) < 1e-10):
            return -1.0  # invalid
        if np.any(cross > 0) and np.any(cross < 0):
            return -1.0  # not convex
        scores = compute_scores(pts)
        return -np.min(scores)

    # Initial guess: regular polygon (zero deviations)
    x0 = np.zeros(n)
    
    # Try local optimization first
    best_score = 0.0
    best_params = x0.copy()
    
    # Evaluate initial
    init_obj = objective(x0)
    if init_obj < 0:
        best_score = -init_obj
    else:
        best_score = 0.0
    
    # Local optimization with multiple starts
    starts = [x0]
    # Add some random perturbations
    rng = np.random.RandomState(42)
    for trial in range(5):
        pert = rng.normal(0, 0.1, n)
        starts.append(pert)
    
    for start in starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, start, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # If we have time, try differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-0.5, 0.5)] * n
            de_result = differential_evolution(objective, bounds, seed=42,
                                              maxiter=200, tol=1e-10,
                                              polish=True, workers=1,
                                              updating='immediate')
            if de_result.fun < 0:
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 30 and best_params is not None:
        try:
            final_result = minimize(objective, best_params, method='L-BFGS-B',
                                   options={'maxiter': 10000, 'ftol': 1e-15})
            if final_result.fun < 0:
                score = -final_result.fun
                if score > best_score:
                    best_score = score
                    best_params = final_result.x.copy()
        except Exception:
            pass
    
    # Construct final polygon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = base_angles + best_params
    # Sort angles to ensure cyclic order
    angles = np.sort(angles)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_centroid = np.sqrt(np.sum(pts ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        # Fallback to regular polygon
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    else:
        pts = pts / mean_dist
    
    return pts

# EVOLVE-BLOCK-END
