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

    # Base grid: 4x4
    base_grid = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3], [1, 3], [2, 3], [3, 3]
    ], dtype=float)
    
    # Center the grid
    base_grid -= base_grid.mean(axis=0)

    def score(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def decode_params(params):
        """
        Decode a vector of parameters into 16 2D points.
        params:
        - params[0]: rotation angle
        - params[1:]: Fourier coefficients for x and y deformations
        """
        theta = params[0]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        pts = base_grid @ rot_matrix.T
        
        # Define Fourier modes to allow smooth distortion
        # We use a few low-frequency modes to deform the grid
        # x(u, v) and y(u, v) where u, v are grid coordinates (0..3)
        u = base_grid[:, 0]
        v = base_grid[:, 1]
        
        # Normalize u, v to [-1, 1] for trig functions
        u_norm = (u - 1.5) / 1.5
        v_norm = (v - 1.5) / 1.5
        
        # Modes:
        # 1. Stretch/Squeeze along axes (already handled by rotation + scale if we had scale, 
        #    but we can add explicit deformation)
        # 2. Shear
        # 3. Curvature (sinusoidal bulge)
        
        # Let's use 4 modes: 
        # m1: sin(pi*u) -> bulge x
        # m2: sin(pi*v) -> bulge y
        # m3: sin(2*pi*u) -> wave x
        # m4: sin(2*pi*v) -> wave y
        
        # params[1] = amp_x1
        # params[2] = amp_y1
        # params[3] = amp_x2
        # params[4] = amp_y2
        
        amp_x1 = params[1]
        amp_y1 = params[2]
        amp_x2 = params[3]
        amp_y2 = params[4]
        
        dx = amp_x1 * np.sin(np.pi * u_norm) + amp_x2 * np.sin(2 * np.pi * u_norm)
        dy = amp_y1 * np.sin(np.pi * v_norm) + amp_y2 * np.sin(2 * np.pi * v_norm)
        
        pts[:, 0] += dx
        pts[:, 1] += dy
        
        # Center again to prevent drift
        pts -= pts.mean(axis=0)
        
        return pts

    def objective(params):
        pts = decode_params(params)
        return score(pts)

    # Bounds for Differential Evolution
    # Rotation: [0, 2*pi]
    # Amplitudes: [-0.5, 0.5] (relative to grid spacing ~1)
    bounds = [
        (0, 2 * np.pi),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5)
    ]
    
    best_score = 1e10
    best_pts = base_grid.copy()
    
    # Run Differential Evolution
    # DE is a global optimizer, suitable for this smaller parameter space
    try:
        if time.time() < deadline - 5:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-10, 
                polish=False, # We will polish manually if needed, or DE is enough
                workers=-1    # Use all cores if possible, but standard DE might be fine
            )
            
            if res.success or res.fun < best_score:
                best_params = res.x
                best_pts = decode_params(best_params)
                best_score = res.fun
                
    except Exception:
        pass

    # Refinement: Use L-BFGS-B on the decoded coordinates if time permits
    # This allows fine-tuning of the exact positions
    if time.time() < deadline - 10:
        current_pts = best_pts.copy()
        
        def local_obj(pts_flat):
            pts = pts_flat.reshape(N, 2)
            return score(pts)
            
        def local_grad(pts_flat):
            pts = pts_flat.reshape(N, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            d_min = d.min()
            d_max = d.max()
            
            if d_min < 1e-12:
                return np.zeros_like(pts_flat)
                
            # Gradient of (d_max/d_min)^2
            # Let f = (d_max/d_min)^2 = d_max^2 * d_min^-2
            # df/dx = 2*d_max*d_min^-2 * (d_max * d(d_max)/dx - d_min * d(d_min)/dx)
            # This is tricky because d_max and d_min are non-smooth.
            # We can use a smooth approximation for the gradient step or just use numerical gradient.
            # Given the time constraint and that DE already found a good structure,
            # a few steps of Nelder-Mead or simple gradient descent on the exact objective
            # might be risky due to non-smoothness.
            # Instead, let's try a few random perturbations and keep the best.
            return None # Skip analytic gradient for now, use stochastic local search

        # Stochastic Local Search
        rng = np.random.default_rng(42)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Small random perturbation
            perturbed = current_pts + rng.normal(0, 0.01, (N, 2))
            sc = score(perturbed)
            if sc < best_score:
                best_score = sc
                best_pts = perturbed

    return best_pts

# EVOLVE-BLOCK-END
