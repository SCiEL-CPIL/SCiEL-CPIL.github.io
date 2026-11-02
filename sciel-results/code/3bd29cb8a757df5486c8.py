import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We target finishing well before that to be safe.
    # Let's use 140s as a hard stop for the main loop, leaving 10s for finalization.
    deadline = start_time + 140.0

    # Precompute indices for all triples for fast area calculation
    triples = list(combinations(range(N), 3))
    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def compute_score(points):
        """
        Compute the raw score: min triangle area / convex hull area.
        """
        if len(points) < 3:
            return 0.0
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 0.0
        except Exception:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        
        # Vectorized cross product calculation for all triangles
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = np.min(areas)
        
        return min_area / hull_area

    def soft_min_objective(params, n_hull, n_inner, beta):
        """
        Objective function for L-BFGS-B.
        params: [r_hull, r_inner, rot_hull, rot_inner]
        Returns negative soft-min score (to be minimized).
        """
        r_hull, r_inner, rot_hull, rot_inner = params
        
        # Constraints: r_hull > 0, r_inner >= 0, r_hull > r_inner (to maintain hierarchy)
        # If constraints are violated, return a large penalty
        if r_hull < 0.1 or r_inner < -0.1 or r_hull < r_inner + 1e-6:
            return 1e6
            
        # Ensure r_inner is non-negative for geometry generation
        r_inner_eff = max(0.0, r_inner)
        
        # Generate points
        points = []
        
        # Outer ring (Hull candidates)
        if n_hull > 0:
            angles_hull = np.linspace(rot_hull, rot_hull + 2 * np.pi, n_hull, endpoint=False)
            pts_hull = np.column_stack((r_hull * np.cos(angles_hull), r_hull * np.sin(angles_hull)))
            points.append(pts_hull)
            
        # Inner ring
        if n_inner > 0:
            angles_inner = np.linspace(rot_inner, rot_inner + 2 * np.pi, n_inner, endpoint=False)
            pts_inner = np.column_stack((r_inner_eff * np.cos(angles_inner), r_inner_eff * np.sin(angles_inner)))
            points.append(pts_inner)
            
        if not points:
            return 1e6
            
        points = np.vstack(points)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 1e6
        except Exception:
            return 1e6

        # Calculate Triangle Areas
        x = points[:, 0]
        y = points[:, 1]
        
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        
        # Soft Min: approximates min(areas)
        # f(x) = -1/beta * log(sum(exp(-beta * x)))
        # We want to MAXIMIZE min(areas)/hull_area
        # So we MINIMIZE - (soft_min(areas) / hull_area)
        
        # Numerical stability for log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
            
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        score = soft_min_area / hull_area
        return -score

    def hard_objective_symmetric(params, n_hull, n_inner):
        """
        Hard objective for Nelder-Mead (no gradient).
        """
        r_hull, r_inner, rot_hull, rot_inner = params
        
        if r_hull < 0.1 or r_inner < -0.1 or r_hull < r_inner + 1e-6:
            return 1e6
            
        r_inner_eff = max(0.0, r_inner)
        
        points = []
        if n_hull > 0:
            angles_hull = np.linspace(rot_hull, rot_hull + 2 * np.pi, n_hull, endpoint=False)
            pts_hull = np.column_stack((r_hull * np.cos(angles_hull), r_hull * np.sin(angles_hull)))
            points.append(pts_hull)
            
        if n_inner > 0:
            angles_inner = np.linspace(rot_inner, rot_inner + 2 * np.pi, n_inner, endpoint=False)
            pts_inner = np.column_stack((r_inner_eff * np.cos(angles_inner), r_inner_eff * np.sin(angles_inner)))
            points.append(pts_inner)
            
        if not points:
            return 1e6
            
        points = np.vstack(points)
        
        score = compute_score(points)
        return -score

    def generate_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner):
        """Utility to generate points from symmetric params."""
        r_inner_eff = max(0.0, r_inner)
        points = []
        if n_hull > 0:
            angles_hull = np.linspace(rot_hull, rot_hull + 2 * np.pi, n_hull, endpoint=False)
            pts_hull = np.column_stack((r_hull * np.cos(angles_hull), r_hull * np.sin(angles_hull)))
            points.append(pts_hull)
        if n_inner > 0:
            angles_inner = np.linspace(rot_inner, rot_inner + 2 * np.pi, n_inner, endpoint=False)
            pts_inner = np.column_stack((r_inner_eff * np.cos(angles_inner), r_inner_eff * np.sin(angles_inner)))
            points.append(pts_inner)
        if not points:
            return np.zeros((N, 2))
        return np.vstack(points)

    best_score = -float('inf')
    best_points = None

    # Splits of 13 points into hull and inner
    # We assume the first n_hull points form the convex hull (outer ring)
    # and the remaining n_inner are inside.
    # Note: For n_hull=1 or 2, the "hull" is a line/point, which is degenerate for area normalization unless we treat the whole set.
    # ConvexHull handles the actual hull. But geometrically, a 1-gon or 2-gon "outer ring" is just points.
    # We should prioritize splits where n_hull is large enough to form a polygon, e.g., >= 3.
    # However, the optimization might find that some "inner" points actually end up on the hull if r_inner is close to r_hull.
    # The parameterization forces them to be on specific circles. 
    # Let's try splits from 3 to 12 for hull, and corresponding inner.
    # (13,0) is a regular 13-gon.
    
    splits = []
    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        splits.append((n_hull, n_inner))
        
    # Also try (12, 1), (11, 2) etc. are covered.
    # What about n_hull=2? A "2-gon" is a line segment. The hull of the set would include the inner points if they are off the line.
    # The symmetric model might be too restrictive for n_hull < 3. Let's stick to n_hull >= 3 for the "outer ring" interpretation.
    # Actually, if n_hull=1, it's just 1 point on the outside, 12 inside. The hull is determined by the 12 inside.
    # The model "Ring 1 is Hull, Ring 2 is Inner" implies Ring 1 is the outer boundary.
    # So n_hull must be >= 3 for a 2D hull area to be non-zero from the outer ring alone.
    # If n_hull < 3, the hull is formed by the inner points (if they are further out? No, r_hull > r_inner).
    # If r_hull > r_inner, and n_hull < 3, the convex hull of the set is determined by the inner points IF the outer points don't extend the hull.
    # But if r_hull > r_inner, the outer points ARE the furthest. So the hull is the convex hull of the outer points.
    # If outer points are collinear (n_hull=2) or single (n_hull=1), the hull area is 0 or degenerate.
    # Thus, we must have n_hull >= 3.

    for n_hull, n_inner in splits:
        if time.time() > deadline:
            break
            
        # Initial guesses
        # Regular polygon intuition:
        # If all on hull (n_inner=0), r_hull=1.
        # If mixed, usually inner radius is smaller.
        r_hull_init = 1.0
        r_inner_init = 0.5 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = np.pi / n_inner if n_inner > 1 else 0.0
        
        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])
        
        # Bounds for L-BFGS-B
        # r_hull: [0.5, 2.0]
        # r_inner: [0.0, r_hull] -> Dynamic bound is hard for L-BFGS-B.
        # We can use [0.0, 1.5] and rely on the penalty in objective if r_inner > r_hull.
        # Or just [0.0, 2.0].
        bounds = [
            (0.5, 2.0),
            (0.0, 1.5),
            (-np.pi, np.pi),
            (-np.pi, np.pi)
        ]
        
        # Phase 1: L-BFGS-B with soft min
        # Try a few different betas (sharpness of the min approximation)
        # Lower beta = smoother, easier to optimize, but less accurate min.
        # Higher beta = sharper, closer to true min, but harder to optimize.
        
        current_x = x0.copy()
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline:
                break
                
            try:
                res = opt.minimize(
                    lambda x, b=beta, nh=n_hull, ni=n_inner: soft_min_objective(x, nh, ni, b),
                    current_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e5: # Valid result
                    current_x = res.x
                    points = generate_symmetric_points(n_hull, n_inner, *current_x)
                    score = compute_score(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
            except Exception:
                pass

        # Phase 2: Nelder-Mead on symmetric params with Hard Objective
        # This helps navigate the non-smooth "min" landscape near the optimum.
        if time.time() > deadline:
            continue
            
        try:
            res = opt.minimize(
                lambda x, nh=n_hull, ni=n_inner: hard_objective_symmetric(x, nh, ni),
                current_x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < 1e5:
                final_x = res.x
                points = generate_symmetric_points(n_hull, n_inner, *final_x)
                score = compute_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass

    # If no valid points found (should not happen), fallback to regular 13-gon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
