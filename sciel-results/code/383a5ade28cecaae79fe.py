"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _fast_objective(x, deadline):
    """
    Fast objective function for 16D vector (8 points).
    Minimizes -log(Score).
    """
    if time.time() > deadline:
        return 1e10
        
    n = 8
    pts = x.reshape(n, 2)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = np.sqrt(dists_sq[mask])
    
    if np.any(dists < 1e-12):
        return 1e10
        
    D = np.max(dists)
    if D < 1e-12:
        return 1e10
        
    sum_log_d = np.sum(np.log(dists))
    log_score = sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0)
    
    return -log_score

def _cs_param_to_points(p):
    """
    Convert 8D parameter vector (4 points) to 16D point vector.
    Enforces central symmetry: points are p, -p.
    p = [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    pts = p.reshape(4, 2)
    # The 8 points are the 4 points and their negations
    full_pts = np.vstack([pts, -pts])
    return full_pts.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    
    # 1. Global Search in Reduced Parameter Space (Central Symmetry Model)
    # Parameters: 4 points (x1, y1, ..., x4, y4)
    # Bounds: [-2, 2] for each coordinate
    
    bounds = [(-2.0, 2.0)] * 8
    
    def cs_objective(p):
        if time.time() > deadline:
            return 1e10
        pts = _cs_param_to_points(p)
        return _fast_objective(pts, deadline)
    
    best_x_param = None
    best_val_param = float('inf')
    
    # Run Differential Evolution
    try:
        de_res = differential_evolution(
            cs_objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        if de_res.fun < best_val_param:
            best_val_param = de_res.fun
            best_x_param = de_res.x
    except Exception:
        pass
    
    # If DE failed or didn't run, fallback to regular octagon params
    if best_x_param is None:
        # Regular octagon: 4 points are (1,0), (1/sqrt(2), 1/sqrt(2)), (0,1), (-1/sqrt(2), 1/sqrt(2))
        # Wait, regular octagon vertices are at angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
        # Central symmetry means we pick 4 of them such that the other 4 are their negations.
        # (1,0) -> (-1,0)
        # (1/sqrt(2), 1/sqrt(2)) -> (-1/sqrt(2), -1/sqrt(2))
        # (0,1) -> (0,-1)
        # (-1/sqrt(2), 1/sqrt(2)) -> (1/sqrt(2), -1/sqrt(2))
        # So the 4 unique points are:
        # (1, 0)
        # (1/sqrt(2), 1/sqrt(2))
        # (0, 1)
        # (-1/sqrt(2), 1/sqrt(2))
        s = 1.0 / np.sqrt(2)
        best_x_param = np.array([1.0, 0.0, s, s, 0.0, 1.0, -s, s])
        
    # Convert best param to points
    best_pts = _cs_param_to_points(best_x_param)
    
    # 2. Local Refinement in Full 16D Space
    # Use L-BFGS-B for speed
    
    def full_objective(x):
        return _fast_objective(x, deadline)
    
    # Try polishing from the symmetric solution
    try:
        res = minimize(
            full_objective,
            best_pts,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-15}
        )
        if res.fun < best_val_param:
            best_val_param = res.fun
            best_pts = res.x
    except Exception:
        pass
    
    # 3. Try a few other structured initializations in 16D and polish
    # e.g. Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Random perturbations of the best found
    rng = np.random.RandomState(42)
    
    for i in range(15):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best
        noise = rng.randn(16) * 0.1
        x_init = best_pts + noise
        
        try:
            res = minimize(
                full_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15}
            )
            if res.fun < best_val_param:
                best_val_param = res.fun
                best_pts = res.x
        except Exception:
            pass
            
    # Also try polishing the regular octagon directly in case the CS model missed it
    try:
        res_oct = minimize(
            full_objective,
            x0_oct,
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-15}
        )
        if res_oct.fun < best_val_param:
            best_val_param = res_oct.fun
            best_pts = res_oct.x
    except Exception:
        pass

    # 4. Final Polish with Powell for high precision
    if time.time() < deadline - 2:
        try:
            res_final = minimize(
                full_objective,
                best_pts,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res_final.fun < best_val_param:
                best_val_param = res_final.fun
                best_pts = res_final.x
        except Exception:
            pass

    best_points = best_pts.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
