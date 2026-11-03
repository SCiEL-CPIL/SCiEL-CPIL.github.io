"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses iterative bisection on radius r and scipy.optimize.least_squares 
    to find a feasible packing for each r.
    """
    n = N
    
    # 1. Generate initial hexagonal lattice points
    # Estimate spacing for n points in unit disk
    # Area ~ pi * 1^2 = pi. Each circle area ~ pi * r^2.
    # n * pi * r^2 <= pi => r <= 1/sqrt(n)
    # Hex packing density ~ 0.9069.
    # Effective area per point ~ pi * r^2 / 0.9069
    # n * pi * r^2 / 0.9069 <= pi => r <= sqrt(0.9069/n)
    # Let's just generate a dense hex grid and pick the best n.
    
    def generate_hex_grid(n_points, max_radius=1.0):
        # Estimate spacing
        # Approximate number of points in hex grid with spacing s in radius R:
        # ~ (2*sqrt(3)/s^2) * R^2 * density?
        # Let's just try a range of spacings.
        best_points = None
        for s in np.linspace(0.03, 0.1, 50):
            # Generate grid
            # Rows and cols
            # For hex grid, row j is offset by s/2 if j is odd.
            # Vertical spacing is s * sqrt(3) / 2
            k = int(1.0 / s) + 2
            xs = np.arange(-k, k + 1) * s
            ys = np.arange(-k, k + 1) * s * np.sqrt(3.0) / 2.0
            xx, yy = np.meshgrid(xs, ys)
            # Apply offset for odd rows (index in ys)
            # Actually, standard hex grid:
            # x = i*s + (j%2)*s/2
            # y = j*s*sqrt(3)/2
            
            # Let's construct points directly
            points = []
            for j in range(-k, k + 1):
                offset = (j % 2) * 0.5 * s
                for i in range(-k, k + 1):
                    x = i * s + offset
                    y = j * s * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= max_radius:
                        points.append([x, y])
            points = np.array(points)
            if len(points) < n_points:
                continue
            
            # Sort by distance to center to get the most central points
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)
            points = points[indices[:n_points]]
            
            # Calculate min distance and min boundary dist
            if len(points) < 2:
                continue
            min_pair_dist = np.min(pdist(points))
            min_boundary_dist = np.min(1.0 - dists[:n_points])
            r_candidate = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if best_points is None or r_candidate > best_points[1]:
                best_points = (points, r_candidate)
                
        if best_points is None:
             # Fallback: simple grid
             xs = np.linspace(-0.9, 0.9, 20)
             ys = np.linspace(-0.9, 0.9, 20)
             points = np.array([[x, y] for x in xs for y in ys])
             points = points[:n_points]
             return points
             
        return best_points[0]

    # Initial guess
    x0 = generate_hex_grid(n).flatten()
    
    def residuals(x, r_target):
        centres = x.reshape(n, 2)
        
        # Pairwise distances
        dists = pdist(centres)
        # Shortfalls for pairs: we want d_ij >= 2*r_target
        # Residual is max(0, 2*r_target - d_ij)
        pair_res = np.maximum(0.0, 2.0 * r_target - dists)
        
        # Boundary distances
        norms = np.linalg.norm(centres, axis=1)
        boundary_dists = 1.0 - norms
        # Shortfalls for boundary: we want 1 - |c_i| >= r_target
        # Residual is max(0, r_target - (1 - |c_i|))
        bound_res = np.maximum(0.0, r_target - boundary_dists)
        
        return np.concatenate([pair_res, bound_res])
    
    # Bisection search for maximum r
    # Lower bound: from initial guess
    min_pair_dist = np.min(pdist(x0.reshape(n, 2)))
    min_boundary_dist = np.min(1.0 - np.linalg.norm(x0.reshape(n, 2), axis=1))
    r_low = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Upper bound: theoretical max is 1/sqrt(n) * sqrt(density) approx
    # For n=300, 1/sqrt(300) ~ 0.0577.
    # Best known is 0.05315.
    # Let's set a safe upper bound.
    r_high = 1.0 / np.sqrt(n) * 1.1 # Slightly above theoretical limit
    
    # Ensure r_low is feasible (it should be, as it's derived from the config)
    # But let's verify with a quick optimization or just assume it's a lower bound.
    # Actually, r_low is the *achieved* radius of the initial guess.
    # So r_low is definitely feasible.
    
    # We want to find the max r such that min ||F(x, r)||^2 is small enough.
    # Threshold for feasibility
    tol = 1e-6
    
    current_x = x0.copy()
    
    # Perform bisection
    # We will try to increase r.
    # If we can't fit r_high, we bisect.
    
    # First, check if we can improve from initial guess.
    # Run optimization for a slightly larger r?
    # Or just bisect between r_low and r_high.
    
    # To save time, we can do a few iterations of bisection.
    # 120s limit. least_squares can be slow.
    # Let's do ~20 bisection steps.
    
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        
        # Try to solve for r_mid starting from current_x
        try:
            res = least_squares(
                residuals, 
                current_x, 
                args=(r_mid,), 
                method='trf', 
                max_nfev=2000,
                xtol=1e-8, 
                ftol=1e-8, 
                gtol=1e-8
            )
            
            # Check if solution is feasible
            # Sum of squared residuals should be close to 0
            cost = np.sum(res.fun ** 2)
            
            if cost < tol:
                # Feasible, so we can try a larger r
                r_low = r_mid
                current_x = res.x
            else:
                # Not feasible, try smaller r
                r_high = r_mid
                # Keep current_x? Or revert?
                # It's better to keep the last known feasible x for warm start.
                # But if we fail, we might have moved away.
                # Let's keep the last feasible x.
                pass
                
        except Exception:
            r_high = r_mid
            
    # After bisection, r_low is the best feasible radius found.
    # Let's do one final optimization at r_low to polish the solution
    # using the last successful x.
    
    # We need to recover the x corresponding to r_low.
    # In the loop, we updated current_x only when feasible.
    # So current_x is the solution for the last feasible r_low.
    
    # Final polish
    try:
        res_final = least_squares(
            residuals, 
            current_x, 
            args=(r_low,), 
            method='trf', 
            max_nfev=5000,
            xtol=1e-10, 
            ftol=1e-10, 
            gtol=1e-10
        )
        final_centres = res_final.x.reshape(n, 2)
        
        # Verify
        dists = pdist(final_centres)
        min_pair_dist = np.min(dists)
        min_boundary_dist = np.min(1.0 - np.linalg.norm(final_centres, axis=1))
        final_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if final_r > 0 and np.isfinite(final_r):
            return final_centres
    except Exception:
        pass
    
    # Fallback: return the last feasible x found during bisection
    # Or the initial guess if bisection failed immediately.
    # current_x should be valid if r_low > 0.
    return current_x.reshape(n, 2)
# EVOLVE-BLOCK-END
