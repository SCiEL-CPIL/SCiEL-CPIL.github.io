"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Define the 16 points as a hexagonal cluster with 3 DOF
    # The cluster consists of a central point, a ring of 6, and a ring of 9.
    # This structure naturally provides high local density (small d_min) and
    # a controlled diameter (d_max).
    
    # Pre-compute angles for the rings
    # Ring 1: 6 points
    theta1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    # Ring 2: 9 points
    theta2 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    
    # Base vectors (unit circle)
    base1 = np.column_stack((np.cos(theta1), np.sin(theta1)))
    base2 = np.column_stack((np.cos(theta2), np.sin(theta2)))
    
    def score_raw(pts):
        """Calculate the squared ratio of max to min pairwise distance."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = sq_dists[iu]
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def objective(params):
        """
        Objective function for differential_evolution.
        params: [aspect_ratio, shear_angle, radial_scale]
        
        aspect_ratio: scales the y-coordinate of the lattice (1.0 is isotropic)
        shear_angle: rotates the lattice (0.0 is aligned with axes)
        radial_scale: scales the outer ring relative to the inner ring
        """
        aspect = params[0]
        shear = params[1]
        r_scale = params[2]
        
        # Create rotation matrix for shear
        c, s = np.cos(shear), np.sin(shear)
        R = np.array([[c, -s], [s, c]])
        
        # Build points
        # Center
        pts = [[0, 0]]
        
        # Ring 1 (radius 1)
        p1 = base1 * 1.0
        # Apply aspect ratio (stretch y)
        p1[:, 1] *= aspect
        # Apply rotation
        p1 = p1 @ R.T
        pts.extend(p1)
        
        # Ring 2 (radius r_scale)
        # In a perfect hex lattice, the second ring radius is sqrt(3) or 2 depending on indexing.
        # Let's assume the "hexagonal number" arrangement where neighbors are at dist 1.
        # The distance from center to ring 1 is 1.
        # The distance from center to ring 2 should be roughly sqrt(3) ~ 1.732 for a tight packing,
        # or 2.0 if it's the next shell.
        # Let's use r_scale to optimize this ratio.
        p2 = base2 * r_scale
        # Apply aspect ratio
        p2[:, 1] *= aspect
        # Apply rotation
        p2 = p2 @ R.T
        pts.extend(p2)
        
        pts = np.array(pts)
        
        # Normalize the configuration to a fixed size to prevent the optimizer from 
        # trivially shrinking or expanding the whole cluster.
        # The ratio is scale-invariant, so we can fix the max distance to 1.
        # However, for numerical stability in DE, it's better to work in a bounded space.
        # Let's normalize such that the max distance is 1.
        # Actually, since the ratio is scale invariant, we don't strictly need to normalize 
        # for the score calculation, but it helps keep the gradient (if we used it) sane.
        # With DE, we just evaluate the score.
        
        return score_raw(pts)

    # 2. Define bounds for the parameters
    # aspect_ratio: 0.5 to 1.5 (symmetry breaking)
    # shear_angle: 0 to pi/6 (hex symmetry allows this range without loss of generality)
    # r_scale: 1.5 to 2.5 (ratio of outer to inner ring radius)
    
    bounds = [
        (0.5, 1.5),  # aspect
        (0.0, np.pi / 6),  # shear
        (1.5, 2.5)   # r_scale
    ]
    
    # 3. Run Differential Evolution
    # DE is robust for this kind of non-convex, low-dimensional problem.
    # We use a tight tolerance and a reasonable number of iterations.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8, 
            polish=True,  # Use L-BFGS-B for final polish
            init='sobol' # Use Sobol for better coverage in low dims
        )
        
        best_params = res.x
        best_score = res.fun
        
    except Exception:
        # Fallback if DE fails
        best_params = [1.0, 0.0, 2.0]
        best_score = objective(best_params)

    # 4. Construct the final points from the best parameters
    aspect = best_params[0]
    shear = best_params[1]
    r_scale = best_params[2]
    
    c, s = np.cos(shear), np.sin(shear)
    R = np.array([[c, -s], [s, c]])
    
    pts = [[0, 0]]
    
    p1 = base1 * 1.0
    p1[:, 1] *= aspect
    p1 = p1 @ R.T
    pts.extend(p1)
    
    p2 = base2 * r_scale
    p2[:, 1] *= aspect
    p2 = p2 @ R.T
    pts.extend(p2)
    
    final_pts = np.array(pts)
    
    # Ensure distinctness (should be guaranteed by the geometry, but check just in case)
    # If any points are too close, perturb them slightly.
    # Given the structure, this is unlikely unless aspect or shear is degenerate.
    
    return final_pts

# EVOLVE-BLOCK-END
