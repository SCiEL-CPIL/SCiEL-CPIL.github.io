# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

N = 100

def _generate_bcc_base(n_points=100):
    """
    Generate a base configuration of n_points from a BCC lattice,
    selecting the points closest to the origin.
    """
    # Integer lattice points in [-2, 2]
    grid_int = np.array(np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3))).T.reshape(-1, 3)
    # Half-integer lattice points in [-1.5, 1.5]
    grid_half = np.array(np.meshgrid(np.arange(-1.5, 2, 1), np.arange(-1.5, 2, 1), np.arange(-1.5, 2, 1))).T.reshape(-1, 3)
    
    all_pts = np.vstack([grid_int, grid_half])
    
    # Center at origin (already centered)
    # Sort by distance to origin
    dists = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(dists)[:n_points]
    pts = all_pts[idx].copy()
    
    # Ensure exact centering (remove any drift from selection)
    pts -= np.mean(pts, axis=0)
    
    return pts

def _evaluate_packing(params, base_pts):
    """
    Evaluate the packing radius for given parameters.
    params: [rx, ry, rz, tx, ty, tz, scale]
    Returns the radius (scalar).
    """
    rx, ry, rz, tx, ty, tz, scale = params
    
    # Rotation
    R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    
    # Transform points
    centers = (base_pts * scale) @ R.T
    centers += np.array([tx, ty, tz])
    
    # Check if inside cube [-0.5, 0.5]
    # If any point is outside, return a negative penalty
    if np.any(centers > 0.5) or np.any(centers < -0.5):
        # Penalty: distance outside
        out = np.maximum(np.maximum(centers - 0.5, 0), np.maximum(-0.5 - centers, 0))
        penalty = np.max(out)
        return -penalty
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Vectorized
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Ensure writable for fill_diagonal
    dists = np.array(dists, copy=True)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _objective(params, base_pts):
    """
    Objective function for minimization (negative radius).
    """
    r = _evaluate_packing(params, base_pts)
    return -r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    Uses a rigid BCC lattice base, optimized by rotation, translation, and scaling.
    """
    total_time_limit = 120.0
    start_time = time.time()
    
    base_pts = _generate_bcc_base(N)
    
    # Initial guess: identity rotation, zero translation, scale ~ 0.1
    # Max scale for BCC 100 pts?
    # The bounding box of the 100 closest BCC pts is roughly 4x4x4 units in lattice space.
    # We need it to fit in 1.0. So scale ~ 1/4 = 0.25?
    # But we also need to account for rotation.
    # Let's start with scale 0.15.
    
    initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15]
    
    # Bounds
    bounds = [
        (-np.pi, np.pi),  # rx
        (-np.pi, np.pi),  # ry
        (-np.pi, np.pi),  # rz
        (-0.5, 0.5),      # tx
        (-0.5, 0.5),      # ty
        (-0.5, 0.5),      # tz
        (0.05, 0.3)       # scale
    ]
    
    best_params = initial_guess
    best_r = _evaluate_packing(initial_guess, base_pts)
    
    # Differential Evolution for global search
    # Use a subset of the time for DE
    de_time_limit = 60.0
    de_start = time.time()
    
    try:
        # DE might be slow if popsize is large.
        # 7D problem. popsize=15 is default.
        # maxiter=100.
        # We check time in callback? No, DE doesn't have a simple time callback.
        # We can run DE with a fixed number of iterations and check time after.
        # Or just run it and hope it finishes.
        # To be safe, we limit maxiter and popsize.
        
        res_de = differential_evolution(
            _objective, 
            bounds, 
            args=(base_pts,),
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            init='sobol' # If supported, else 'latinhypercube'
        )
        
        if res_de.success or res_de.fun < _objective(initial_guess, base_pts):
            best_params = res_de.x
            best_r = -res_de.fun
            
    except Exception as e:
        # If DE fails, fall back to local search from initial guess
        pass
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Local refinement using Nelder-Mead
    if remaining > 10.0:
        try:
            res_nm = minimize(
                _objective,
                best_params,
                args=(base_pts,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_nm.fun < _objective(best_params, base_pts):
                best_params = res_nm.x
                best_r = -res_nm.fun
                
        except Exception:
            pass
    
    # Final evaluation and construction
    r_final = _evaluate_packing(best_params, base_pts)
    
    # If the best found is invalid (negative), try a few random restarts quickly
    if r_final < 0:
        # Simple random search
        for _ in range(100):
            if time.time() - start_time > total_time_limit - 5:
                break
            rand_params = np.random.uniform(bounds[0][0], bounds[0][1], size=3)
            rand_trans = np.random.uniform(bounds[3][0], bounds[3][1], size=3)
            rand_scale = np.random.uniform(bounds[6][0], bounds[6][1])
            p = np.concatenate([rand_params, rand_trans, [rand_scale]])
            r_rand = _evaluate_packing(p, base_pts)
            if r_rand > best_r:
                best_r = r_rand
                best_params = p
    
    # Extract centers
    rx, ry, rz, tx, ty, tz, scale = best_params
    R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    centers = (base_pts * scale) @ R.T
    centers += np.array([tx, ty, tz])
    
    # Clip to ensure validity
    centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centers
# EVOLVE-BLOCK-END
