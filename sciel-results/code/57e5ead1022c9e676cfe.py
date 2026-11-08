import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    
    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            # Center points to improve numerical stability
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def objective(points_flat):
        """
        Objective function for optimization (minimization).
        Returns negative score (min triangle area / hull area).
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Define bounds for the optimization.
    # Since the problem is scale-invariant (we normalize by hull area), 
    # we can constrain the points to a reasonable box to help the optimizer.
    # The convex hull will be normalized internally, but starting within a 
    # bounded region prevents the optimizer from wandering off to infinity 
    # or creating degenerate configurations with huge coordinates.
    # A box of [-10, 10] is sufficient for 13 points.
    bounds = [(-10, 10)] * (2 * N)

    # Differential Evolution is a global optimizer.
    # We set popsize=20 (20 * 26 = 520 candidates per generation).
    # maxiter=1000 allows for thorough exploration.
    # tol=1e-10 for precision.
    # seed=42 for reproducibility.
    
    # Check time before starting DE
    if time.time() > start_time + time_limit - 5:
        # Fallback if no time
        # Use a regular polygon as a safe fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    try:
        res = opt.differential_evolution(
            objective,
            bounds,
            maxiter=1000,
            popsize=20,
            tol=1e-10,
            seed=42,
            polish=True,  # Use L-BFGS-B for final polish within DE
            init='sobol', # Use Sobol sequence for better initial population coverage
            update_interval=100 # Check time periodically? No, DE doesn't support callbacks easily. 
                                # We rely on the time limit check afterwards or a tight maxiter.
        )
        
        best_points = res.x.reshape(-1, 2)
        
    except Exception as e:
        # Fallback to a symmetric configuration if DE fails or times out unexpectedly
        # We'll try a few quick local optimizations on symmetric starts
        best_points = None
        best_score = -float('inf')
        
        # Generate a few symmetric starts
        for r_inner in [0.0, 0.5]:
            # 12 outer + 1 inner (if r_inner > 0) or 13 outer
            if r_inner == 0.0:
                angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
                pts = np.column_stack([np.cos(angles), np.sin(angles)])
            else:
                angles_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
                inner = np.array([[0.0, 0.0]])
                pts = np.vstack([outer, inner])
                
            # Local polish
            try:
                res_local = opt.minimize(
                    objective, 
                    pts.flatten(), 
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                score = -res_local.fun
                if score > best_score:
                    best_score = score
                    best_points = res_local.x.reshape(-1, 2)
            except Exception:
                pass
        
        if best_points is None:
             angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
             best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Final local polish with Powell to ensure we are at a local maximum
    # Check time
    if time.time() < start_time + time_limit - 2:
        try:
            res_polish = opt.minimize(
                objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            # Verify the polish didn't make it worse (shouldn't happen, but safe)
            score_before = compute_score(best_points)
            score_after = -res_polish.fun
            if score_after > score_before:
                best_points = res_polish.x.reshape(-1, 2)
        except Exception:
            pass

    return best_points
