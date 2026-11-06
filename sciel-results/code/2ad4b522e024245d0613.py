import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 8
    
    def compute_score_from_gaps(gaps):
        """
        Compute the score given angular gaps between 8 points on a circle.
        The points are on the unit circle. The diameter is the max chord length.
        We scale the set to diameter 2.
        Score = (Product of all pairwise distances) / (8^8)
        """
        # Normalize gaps to sum to 2*pi
        s = np.sum(gaps)
        if s < 1e-10:
            return -np.inf
        gaps = gaps / s * (2 * np.pi)
        
        # Cumulative angles to get positions on unit circle
        # Start at angle 0
        angles = np.zeros(n)
        angles[1:] = np.cumsum(gaps[:-1])
        
        # Coordinates on unit circle
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.column_stack((x, y))
        
        # Calculate pairwise distances
        # dists[i, j] = ||pts[i] - pts[j]||
        # Using vectorized calculation
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        
        # Get upper triangle (i < j)
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.any(d < 1e-10):
            return -np.inf
            
        max_d = np.max(d)
        if max_d < 1e-10:
            return -np.inf
        
        # Scale to diameter 2
        scale = 2.0 / max_d
        d_scaled = d * scale
        
        # Product of distances
        # log sum for numerical stability
        log_prod = np.sum(np.log(d_scaled))
        
        # Score = exp(log_prod) / n^n
        # log_score = log_prod - n * log(n)
        log_score = log_prod - n * np.log(n)
        
        return log_score

    def objective(gaps):
        """
        Objective function to minimize (negative log score).
        """
        # Ensure gaps are positive
        if np.any(gaps <= 0):
            return 1e10
        log_s = compute_score_from_gaps(gaps)
        if np.isinf(log_s):
            return 1e10
        return -log_s

    def get_points_from_gaps(gaps):
        """
        Reconstruct points from optimal gaps and scale to diameter 2.
        """
        s = np.sum(gaps)
        gaps = gaps / s * (2 * np.pi)
        
        angles = np.zeros(n)
        angles[1:] = np.cumsum(gaps[:-1])
        
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.column_stack((x, y))
        
        # Find max distance
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        max_d = np.max(d)
        
        if max_d > 1e-10:
            pts = pts * (2.0 / max_d)
            
        return pts

    # Strategy:
    # 1. Global Search using Differential Evolution over 7 angular gaps.
    #    Bounds: Each gap must be positive.
    #    Since sum is 2*pi, and we have 8 points, average gap is pi/4.
    #    We can bound gaps in [0.01, 2*pi - 0.01*7] roughly, but DE handles bounds better if symmetric.
    #    Let's use bounds [0.01, 2.0] for each gap? No, sum must be 2*pi ~ 6.28.
    #    If one gap is large, others must be small.
    #    Better bounds: [0.05, 2.5] ? 
    #    Actually, DE normalizes internally or works in the box.
    #    Let's use bounds [0.01, 3.0] for each of the 7 gaps.
    #    The 8th gap is determined by 2*pi - sum(first 7). We must ensure it's positive.
    #    To handle the constraint that sum(gaps) < 2*pi, we can optimize 7 gaps and check the 8th.
    #    However, DE requires a fixed dimension. 
    #    Let's optimize 8 gaps but normalize them in the objective. 
    #    This maps the 8-dim box to the simplex.
    #    Bounds: [0.01, 10.0] for all 8 gaps.
    
    bounds = [(0.01, 10.0)] * 8
    
    # Regular octagon gaps: pi/4 ~ 0.785
    # DE will explore the space.
    
    # Use a seed for reproducibility in DE
    np.random.seed(42)
    
    best_log_score = -np.inf
    best_gaps = np.full(8, np.pi/4)
    
    # Run Differential Evolution
    # maxiter: 100, popsize: 15
    # tol: 1e-6
    # time budget: ~30-60s
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False # We will polish manually to save time and control it
        )
        if res_de.fun < 1e9:
            best_gaps = res_de.x
            best_log_score = -res_de.fun
    except Exception as e:
        # Fallback to grid/random if DE fails
        pass

    # Check time
    if time.time() > deadline - 20:
        # Not enough time for refinement, return current best
        pts = get_points_from_gaps(best_gaps)
        return [[float(x), float(y)] for x, y in pts]

    # Local Refinement using Nelder-Mead
    # We optimize the 8 gaps.
    # Initial guess from DE result.
    
    def obj_8d(gaps):
        return objective(gaps)
    
    # Try refining from DE result
    try:
        res_nm = minimize(
            obj_8d, 
            best_gaps, 
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
        )
        if res_nm.fun < -best_log_score:
            best_gaps = res_nm.x
            best_log_score = -res_nm.fun
    except Exception:
        pass

    # Try a few random restarts with local refinement if time permits
    # The space of gap configurations is 7-dimensional (simplex).
    # Random points in the simplex can be generated by sampling 7 exponential variables and normalizing.
    
    if time.time() < deadline - 10:
        for i in range(10):
            # Generate random gaps
            # Sample from Dirichlet distribution (alpha=1 is uniform on simplex)
            # We need 8 gaps summing to 2*pi.
            # Sample 8 random numbers, normalize, multiply by 2*pi.
            raw_gaps = np.random.exponential(1.0, size=8)
            gaps_init = raw_gaps / np.sum(raw_gaps) * (2 * np.pi)
            
            try:
                res_r = minimize(
                    obj_8d, 
                    gaps_init, 
                    method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100}
                )
                if res_r.fun < -best_log_score:
                    best_gaps = res_r.x
                    best_log_score = -res_r.fun
            except Exception:
                pass
            
            if time.time() > deadline - 5:
                break

    # Final Construction
    pts = get_points_from_gaps(best_gaps)
    
    # Ensure distinct points and valid floats
    # The normalization ensures they are on unit circle, scaled to diameter 2.
    # Check for duplicates (should not happen with positive gaps)
    
    return [[float(x), float(y)] for x, y in pts]
