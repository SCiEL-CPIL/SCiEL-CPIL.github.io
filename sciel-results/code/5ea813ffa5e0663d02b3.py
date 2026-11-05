import numpy as np
import time
from scipy.optimize import minimize
import math

def _log_score(points):
    """
    Compute log(Delta) for a set of points.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    Delta = prod_{i!=j} (d_ij / D) where D is the diameter.
    log(Delta) = sum_{i!=j} log(d_ij) - 56 * log(D)
    """
    points = np.array(points)
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return -np.inf
        
    D = np.max(valid_dists)
    if D <= 0:
        return -np.inf
        
    return np.sum(np.log(valid_dists)) - 56 * np.log(D)

def _score(points):
    """
    Compute the actual score Delta / 8^8.
    """
    log_val = _log_score(points)
    if not np.isfinite(log_val):
        return 0.0
    return np.exp(log_val - 8 * np.log(8))

def _poly_to_points(coeffs):
    """
    Convert polynomial coefficients to points.
    coeffs: [1, c_{n-1}, ..., c_0]
    Returns: (n, 2) array of real coordinates.
    """
    # coeffs[0] is 1 (monic)
    # Use numpy roots
    try:
        roots = np.roots(coeffs)
    except Exception:
        return None
    
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-6]
    
    if len(real_roots) < 8:
        # If we don't have 8 distinct real roots, we might have complex roots.
        # The problem requires 8 planar points, which implies we need 8 real distinct points?
        # "Eight planar points" usually means points in R^2.
        # If roots are complex, they come in conjugate pairs. 
        # The distance between z and z* is 2|Im(z)|.
        # However, the task asks for 8 points [x, y]. 
        # If we use complex roots, we get 8 complex numbers. 
        # Do we map them to 8 points in R^2? 
        # Yes, z_k = x_k + i y_k.
        # The distance between z_i and z_j is |z_i - z_j|.
        # This is exactly the distance in the complex plane.
        # So we can just use the real and imaginary parts of the 8 roots.
        # We need to ensure they are distinct.
        pass
        
    points = np.column_stack([roots.real, roots.imag])
    
    # Check distinctness
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
    if np.min(dists[~np.eye(8, dtype=bool)]) < 1e-5:
        return None
        
    return points

def _neg_smooth_log_score_coeffs(x, beta=100.0):
    """
    Objective for coefficient space optimization.
    x: [c_7, c_6, ..., c_0] (8 coefficients, c_8=1 fixed)
    """
    coeffs = np.concatenate(([1.0], x))
    points = _poly_to_points(coeffs)
    
    if points is None:
        return 1e10
        
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 1e-10):
        return 1e10
        
    log_dists = np.log(valid_dists)
    sum_log_dists = np.sum(log_dists)
    
    # Smooth max of log(dists)
    max_log_dist = (1.0 / beta) * np.log(np.sum(np.exp(beta * log_dists)))
    
    log_delta = sum_log_dists - 56 * max_log_dist
    
    if not np.isfinite(log_delta):
        return 1e10
        
    return -log_delta

def _neg_smooth_log_score_coords(x_flat, beta=100.0):
    """
    Objective for coordinate optimization with smooth diameter constraint.
    x_flat: 16 floats (8 points * 2 coords)
    """
    points = x_flat.reshape(8, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 1e-10):
        return 1e10
        
    log_dists = np.log(valid_dists)
    sum_log_dists = np.sum(log_dists)
    
    # Smooth max of log(dists)
    max_log_dist = (1.0 / beta) * np.log(np.sum(np.exp(beta * log_dists)))
    
    # log(Delta) = sum_log_dists - 56 * max_log_dist
    log_delta = sum_log_dists - 56 * max_log_dist
    
    if not np.isfinite(log_delta):
        return 1e10
        
    return -log_delta

def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _coeffs_from_points(points):
    """
    Get monic polynomial coefficients from points (complex roots).
    """
    roots = points[:, 0] + 1j * points[:, 1]
    # np.poly returns [1, c_{n-1}, ..., c_0]
    coeffs = np.poly(roots)
    # Return c_{n-1}...c_0
    return coeffs[1:]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_points = _regular_octagon()
    
    # Update best with regular octagon
    s = _score(best_points)
    if s > best_score:
        best_score = s
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Optimize in coefficient space (smooth landscape for discriminant).
    # 2. Extract points.
    # 3. Polish in coordinate space.
    
    # Initial coefficients from regular octagon
    c0 = _coeffs_from_points(_regular_octagon())
    
    # Run optimization in coefficient space
    while time.time() < deadline:
        # Add noise to start
        c_init = c0 + rng.normal(0, 0.5, 8)
        
        try:
            # Use Nelder-Mead in coeff space
            res_coeff = minimize(_neg_smooth_log_score_coeffs, c_init, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res_coeff.fun < 1e9:
                c_opt = res_coeff.x
                points_opt = _poly_to_points(np.concatenate(([1.0], c_opt)))
                
                if points_opt is not None:
                    # Polish in coordinate space
                    x0 = points_opt.flatten()
                    
                    best_local_score = 0
                    best_local_pts = points_opt
                    
                    # Try a few betas
                    for beta in [50.0, 100.0, 200.0]:
                        try:
                            res_coord = minimize(_neg_smooth_log_score_coords, x0, method='L-BFGS-B',
                                                 args=(beta,), options={'maxiter': 500, 'gtol': 1e-8})
                            if res_coord.fun < 1e9:
                                pts_polished = res_coord.x.reshape(8, 2)
                                s_polished = _score(pts_polished)
                                if s_polished > best_local_score:
                                    best_local_score = s_polished
                                    best_local_pts = pts_polished
                        except Exception:
                            pass
                    
                    if best_local_score > best_score:
                        best_score = best_local_score
                        best_points = best_local_pts
                        
        except Exception:
            pass
        
        # Also try direct coordinate optimization from diverse starts
        if time.time() < deadline:
            # Generate a diverse initial point
            r = rng.random()
            if r < 0.5:
                init_pts = _regular_octagon() + rng.normal(0, 0.1, (8, 2))
            else:
                # Random points in disk
                radii = rng.uniform(0.5, 1.5, 8)
                angles = rng.uniform(0, 2 * np.pi, 8)
                init_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            x0 = init_pts.flatten()
            try:
                res_nm = minimize(_neg_smooth_log_score_coords, x0, method='Nelder-Mead',
                                  args=(100.0,), options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res_nm.fun < 1e9:
                    x_nm = res_nm.x
                    res_lbfgs = minimize(_neg_smooth_log_score_coords, x_nm, method='L-BFGS-B',
                                         args=(200.0,), options={'maxiter': 100, 'gtol': 1e-7})
                    
                    if res_lbfgs.fun < 1e9:
                        pts_final = res_lbfgs.x.reshape(8, 2)
                        s_final = _score(pts_final)
                        if s_final > best_score:
                            best_score = s_final
                            best_points = pts_final
            except Exception:
                pass

    return best_points.tolist()

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_score(np.array(pts))}")
