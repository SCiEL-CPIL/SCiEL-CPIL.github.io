"""
Strategy: Low-rank sinusoidal deformation of a 4x4 grid optimized by Differential Evolution.
Instead of optimizing 32 independent coordinates (which allows points to collapse or spread
irregularly), we represent the 16 points as a deformed 4x4 grid. The deformation is
parameterized by a small number of global geometric parameters (amplitudes of sinusoidal
modes, aspect ratio, rotation). This low-dimensional parameter space (approx 10-12 dims)
inherently preserves local neighborhood structures, preventing the "collapse" failures
common in high-dimensional unconstrained searches. We use SciPy's Differential Evolution
(DE) to explore this bounded, lower-dimensional space robustly, followed by a final
polishing step with L-BFGS-B on a smooth surrogate objective.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Base grid: 4x4 points centered at origin, scaled to unit variance roughly
    # Coordinates: -1.5, -0.5, 0.5, 1.5
    base_grid = np.array([
        [-1.5, -1.5], [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
        [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
        [-1.5, 0.5], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5],
        [-1.5, 1.5], [-0.5, 1.5], [0.5, 1.5], [1.5, 1.5]
    ])
    
    # Precompute indices for grid rows and cols for deformation
    # Row index: 0,1,2,3 for y-coord; Col index: 0,1,2,3 for x-coord
    grid_rows = np.repeat([0, 1, 2, 3], 4)
    grid_cols = np.tile([0, 1, 2, 3], 4)
    
    # Normalize base grid to have unit norm for consistent scaling
    base_grid = base_grid - np.mean(base_grid, axis=0)
    base_norm = np.linalg.norm(base_grid)
    if base_norm > 1e-9:
        base_grid = base_grid / base_norm

    def decode_params(params):
        """
        Decode low-dimensional parameters into 16 2D points.
        params = [
            a1, b1, c1, d1,  # Amplitudes for 4 sinusoidal deformation modes (x, y, x, y?)
            aspect,           # Aspect ratio (x-scale relative to y)
            rot,              # Global rotation angle
            scale             # Global scale (should be 1 if we normalize later, but keep for flexibility)
        ]
        
        Let's use a more structured deformation:
        x_deform = a1 * sin(pi * y_grid / 3) + b1 * cos(pi * y_grid / 3)
        y_deform = c1 * sin(pi * x_grid / 3) + d1 * cos(pi * x_grid / 3)
        
        Where x_grid and y_grid are the normalized grid indices (-1.5 to 1.5).
        """
        a1, b1, c1, d1, aspect, rot, scale = params
        
        # Grid coordinates (already normalized in base_grid)
        x_base = base_grid[:, 0]
        y_base = base_grid[:, 1]
        
        # Deformation fields
        # Using grid indices for smoother phase
        # Map grid indices 0..3 to -1..1 for sine/cos arguments
        idx_x = grid_cols / 3.0 - 0.5  # -0.5, 0.16, 0.83, 1.5? No, let's use -1 to 1
        # Let's just use the base coordinates themselves for the deformation argument, scaled
        arg_x = x_base * 2.0
        arg_y = y_base * 2.0
        
        # Apply deformations
        dx = a1 * np.sin(arg_y) + b1 * np.cos(arg_y)
        dy = c1 * np.sin(arg_x) + d1 * np.cos(arg_x)
        
        pts = base_grid.copy()
        pts[:, 0] += dx
        pts[:, 1] += dy
        
        # Apply aspect ratio
        pts[:, 0] *= aspect
        
        # Apply rotation
        c, s = np.cos(rot), np.sin(rot)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = pts @ rot_mat.T
        
        # Apply scale
        pts *= scale
        
        # Center and normalize to unit norm to remove global translation/scale redundancy
        pts -= np.mean(pts, axis=0)
        norm = np.linalg.norm(pts)
        if norm > 1e-9:
            pts = pts / norm
        else:
            pts = base_grid.copy() # Fallback
            
        return pts

    def score_raw(pts):
        """Calculate (d_max / d_min)^2"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = sq_dists[iu]
        
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-12:
            return 1e10
            
        return d_max_sq / d_min_sq

    def objective(params):
        """Objective function for optimization. Returns score."""
        try:
            pts = decode_params(params)
            # Check for distinct points
            # If any two points are too close, penalize heavily
            # The raw score handles this via d_min -> 0, but we can add a small penalty
            # to help gradients if we were using them. DE is derivative-free, so raw score is fine.
            return score_raw(pts)
        except Exception:
            return 1e10

    # Differential Evolution bounds
    # Amplitudes: small, to avoid destroying the grid structure
    # Aspect: 0.5 to 2.0
    # Rot: -pi to pi
    # Scale: 0.5 to 2.0 (though normalized later, helps exploration)
    
    bounds = [
        (-0.2, 0.2),  # a1
        (-0.2, 0.2),  # b1
        (-0.2, 0.2),  # c1
        (-0.2, 0.2),  # d1
        (0.5, 2.0),   # aspect
        (-np.pi, np.pi), # rot
        (0.5, 2.0)    # scale
    ]
    
    best_score = float('inf')
    best_pts = None
    
    # Check time before starting
    if time.time() > deadline - 10.0:
        return base_grid

    try:
        # Run Differential Evolution
        # maxiter: 200 is usually enough for low dim
        # popsize: 10-20
        # tol: 1e-5
        # seed: 42 for reproducibility, or None for random
        # We have ~130s. DE is relatively fast in low dims.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            init='sobol',
            # Polish with L-BFGS-B? DE has a polish option, but it uses BFGS.
            # Let's do manual polish for better control.
            polish=False 
        )
        
        best_params = result.x
        best_score = result.fun
        best_pts = decode_params(best_params)
        
    except Exception:
        # Fallback to base grid if DE fails
        best_pts = base_grid.copy()
        best_score = score_raw(best_pts)

    # Check time before polishing
    if time.time() < deadline - 20.0:
        # Polish with L-BFGS-B on a smooth surrogate?
        # Or just refine the raw score with a smaller DE or simple gradient descent?
        # Since raw score is non-smooth (min/max), L-BFGS-B might struggle.
        # Let's use a smooth approximation for polishing:
        # Soft max/min using log-sum-exp
        beta = 50.0
        
        def smooth_objective(params):
            pts = decode_params(params)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            iu = np.triu_indices(N, 1)
            d_sq = sq_dists[iu]
            
            # Soft max: log(sum(exp(beta*x))) / beta
            # Soft min: -log(sum(exp(-beta*x))) / beta
            # Ratio: soft_max / soft_min
            # Score: (soft_max / soft_min)^2
            
            # To avoid overflow, shift by max/min
            d_max = np.max(d_sq)
            d_min = np.min(d_sq)
            
            if d_min < 1e-9:
                return 1e10
                
            soft_max = d_max + np.log(np.sum(np.exp(beta * (d_sq - d_max)))) / beta
            soft_min = d_min - np.log(np.sum(np.exp(-beta * (d_sq - d_min)))) / beta
            
            if soft_min < 1e-9:
                return 1e10
                
            ratio = soft_max / soft_min
            return ratio * ratio

        try:
            res_polish = minimize(
                smooth_objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res_polish.fun < best_score:
                # Verify with raw score
                candidate_pts = decode_params(res_polish.x)
                candidate_score = score_raw(candidate_pts)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts
                    
        except Exception:
            pass

    # Final check
    if best_pts is None:
        best_pts = base_grid.copy()
        
    return best_pts

# EVOLVE-BLOCK-END
