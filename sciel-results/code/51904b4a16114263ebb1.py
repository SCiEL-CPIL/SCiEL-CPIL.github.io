"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def score(pts):
        """Calculate the raw score Delta/n^n for a set of 8 points."""
        if len(pts) != 8:
            return 0.0
        pts = np.array(pts)
        
        # Calculate all pairwise distances
        # Use vectorized distance calculation for speed
        # pts shape: (8, 2)
        # diff shape: (8, 8, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Exclude self-distances (diagonal)
        # We need the product of distances for all ordered pairs i != j
        # There are 8*7 = 56 such pairs.
        # The matrix `dists` has 0s on the diagonal.
        # We can take the product of all elements except the diagonal.
        # Or simpler: flatten and remove diagonal indices.
        idx = np.arange(8)
        dists[idx, idx] = 1.0 # Set diagonal to 1 so it doesn't affect product
        
        # Diameter is the max distance
        d = np.max(dists)
        if d == 0:
            return 0.0
        
        # Scale to diameter 2
        scaled_dists = dists * (2.0 / d)
        
        # Product of distances
        # log-sum-exp trick is safer, but for n=8, 56 terms, product is usually fine.
        # However, if any distance is 0 (duplicate points), log(0) = -inf.
        # We assume distinct points from the parameterization.
        
        log_prod = np.sum(np.log(scaled_dists))
        prod = np.exp(log_prod)
        
        # Score is Delta / n^n
        # n=8, so n^n = 8^8 = 16777216
        score_val = prod / (8.0 ** 8)
        
        return score_val

    def generate_points(params):
        """
        Generate 8 points from 3 parameters: r1, r2, theta.
        Square 1: radius r1, rotation 0
        Square 2: radius r2, rotation theta
        Points are distinct if r1 != r2 or theta is not a multiple of pi/2.
        """
        r1, r2, theta = params
        # Ensure radii are positive
        r1 = abs(r1)
        r2 = abs(r2)
        
        # Square 1 vertices (angles: 0, pi/2, pi, 3pi/2)
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2 vertices (angles: theta, theta+pi/2, theta+pi, theta+3pi/2)
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        # Combine
        pts = np.vstack([pts1, pts2])
        
        # Check for duplicates
        # If r1 == r2 and theta is multiple of pi/2, we have duplicates.
        # The score function handles 0 distances poorly (log(0)).
        # We should penalize this in the objective or avoid it in search.
        
        return pts

    def objective(params):
        """Objective function for minimization (negative score)."""
        pts = generate_points(params)
        
        # Check for near-duplicates to avoid log(0) issues
        # Calculate min distance between distinct points
        # For efficiency, we can just check if any distance is very small
        # But since we are doing a grid search, we can just skip if score is 0 or NaN
        s = score(pts)
        if not np.isfinite(s) or s == 0:
            return 1e10
        return -s

    # Search Strategy:
    # 1. Coarse Grid Search over (r1, r2, theta)
    # 2. Local Refinement from best grid points
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Regular octagon fallback
    # Regular octagon corresponds to r1 = r2 = R and theta = pi/8 (45 deg)
    # Let's initialize best with regular octagon
    reg_oct = np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
    best_score = score(reg_oct)
    best_points = reg_oct.copy()
    
    # Grid Search Parameters
    # r1, r2 in [0.1, 2.0] (scale doesn't matter for score, but helps optimization)
    # theta in [0, pi/2] (symmetry of squares)
    
    # Coarse grid
    r1_vals = np.linspace(0.1, 2.0, 20)
    r2_vals = np.linspace(0.1, 2.0, 20)
    theta_vals = np.linspace(0, np.pi/2, 20)
    
    best_grid_score = 0.0
    best_grid_params = None
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for theta in theta_vals:
                if time.time() > deadline - 30: # Keep time for refinement
                    break
                params = [r1, r2, theta]
                s = score(generate_points(params))
                if s > best_grid_score:
                    best_grid_score = s
                    best_grid_params = params
    
    if best_grid_params is not None and best_grid_score > best_score:
        best_score = best_grid_score
        best_params = best_grid_params
        best_points = generate_points(best_params)

    # Local Refinement
    # Start from the best grid point and the regular octagon equivalent
    # Regular octagon in this parameterization: r1=1, r2=1, theta=pi/8
    start_candidates = []
    if best_params is not None:
        start_candidates.append(best_params)
    start_candidates.append([1.0, 1.0, np.pi/8]) # Regular octagon
    
    # Also try some random starts in the neighborhood of the best grid point
    if best_params is not None:
        for _ in range(5):
            noise = np.random.normal(0, 0.1, 3)
            start_candidates.append(best_params + noise)
    
    for start in start_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B for local optimization
            # Bounds: r1, r2 > 0, theta in [0, pi]
            # We use abs(r) in generate_points, so bounds can be [-2, 2] but abs makes gradient tricky.
            # Better to restrict bounds to positive and handle theta carefully.
            # Let's use bounds: r1 in [0.01, 2], r2 in [0.01, 2], theta in [0, pi]
            bounds = [(0.01, 2.0), (0.01, 2.0), (0, np.pi)]
            
            res_local = minimize(
                objective, 
                start, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.success or res_local.fun < objective(start):
                cand_params = res_local.x
                cand_pts = generate_points(cand_params)
                cand_score = score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_params = cand_params
                    best_points = cand_pts.copy()
                    
        except Exception:
            pass

    # Final check
    if time.time() > deadline:
        if best_points is not None:
            return best_points.tolist()
        return best_points.tolist()

    if best_points is None:
        return reg_oct.tolist()

    return best_points.tolist()

# EVOLVE-BLOCK-END
