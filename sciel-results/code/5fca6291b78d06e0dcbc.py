import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0

    # Fixed angles for regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def _get_points(r):
        return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)

    def _normalize(pts):
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        dists = np.linalg.norm(pts_c, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return pts_c, False
        return pts_c / mean_dist, True

    def _is_strictly_convex(pts):
        # Check cross products
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            p2 = pts[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross <= 1e-9: # Must be strictly positive for CCW
                return False
        return True

    def _score_polygon(pts):
        norm_pts, valid_norm = _normalize(pts)
        if not valid_norm:
            return 0.0
        if not _is_strictly_convex(norm_pts):
            return 0.0
            
        # Vectorized score calculation
        diff = norm_pts[:, np.newaxis, :] - norm_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        sorted_dists = np.sort(dists, axis=1)
        
        # d_0 check
        if np.any(sorted_dists[:, 0] < 0.05):
             # We can still compute others, but these will be 0
             pass
             
        diffs = np.diff(sorted_dists, axis=1)
        k = np.argmin(diffs, axis=1)
        
        rows = np.arange(n)
        d_k = sorted_dists[rows, k]
        d_k1 = sorted_dists[rows, k + 1]
        D = (d_k + d_k1) / 2.0
        
        if np.any(D < 0.1):
             pass
             
        idx0 = np.where(k == 0, 0, np.where(k == n - 3, n - 5, k - 1)) # n-3 is 17
        idx1 = np.where(k == 0, 1, np.where(k == n - 3, n - 4, k))
        idx2 = np.where(k == 0, 2, np.where(k == n - 3, n - 3, k + 1))
        idx3 = np.where(k == 0, 3, np.where(k == n - 3, n - 2, k + 2))
        
        four_d = np.stack([
            sorted_dists[rows, idx0],
            sorted_dists[rows, idx1],
            sorted_dists[rows, idx2],
            sorted_dists[rows, idx3]
        ], axis=1)
        
        D_expanded = D[:, np.newaxis]
        ratios = np.maximum(four_d / D_expanded, D_expanded / four_d)
        raw = np.mean(ratios, axis=1)
        
        scores = np.minimum(1.0, 1.0 / raw)
        
        # Apply penalties
        scores[sorted_dists[:, 0] < 0.05] = 0.0
        scores[D < 0.1] = 0.0
        
        return np.min(scores)

    def _objective(r):
        pts = _get_points(r)
        score = _score_polygon(pts)
        # We want to maximize score, so minimize -score
        # If score is 0 (invalid), we want a high penalty
        # Max score is 1. Min score is 0.
        # If valid, return -score (range [-1, 0])
        # If invalid, return 1.0 (penalty)
        if score <= 0:
            return 1.0
        return -score

    best_score = 0.0
    best_r = np.ones(n)
    
    # Multi-restart
    np.random.seed(42)
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Generate initial radii
        # Mix of regular and perturbed
        if i == 0:
            r_init = np.ones(n)
        else:
            # Add sine wave perturbation
            freq = np.random.randint(1, 5)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.05, 0.2)
            r_init = 1.0 + amp * np.sin(freq * angles + phase)
            
        try:
            res = minimize(
                _objective, 
                r_init, 
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if time.time() > deadline - 10:
                break
                
            r_res = res.x
            score_res = _score_polygon(_get_points(r_res))
            
            if score_res > best_score:
                best_score = score_res
                best_r = r_res
                
        except Exception as e:
            pass

    final_pts = _get_points(best_r)
    # Final normalization is done in evaluator, but we return the points
    # The evaluator will normalize. We should return the points such that
    # when normalized, they are good.
    # Actually, the function returns the raw points. The evaluator normalizes them.
    # So we can return the points generated from best_r.
    
    return final_pts
