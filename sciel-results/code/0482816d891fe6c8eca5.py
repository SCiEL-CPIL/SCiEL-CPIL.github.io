import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    def compute_log_score(params):
        """Compute log of normalized product of pairwise distances.
        We maximize log_prod - 28*log(D) where D is the diameter.
        The constant -8*log(8) is dropped for optimization purposes."""
        x = params[:7]
        y = params[7:14]
        x7 = -np.sum(x)
        y7 = -np.sum(y)
        xs = np.append(x, x7)
        ys = np.append(y, y7)
        points = np.column_stack((xs, ys))
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(8, k=1)
        pair_dists_sq = dists_sq[iu]
        
        min_d = np.min(pair_dists_sq)
        if min_d < 1e-16:
            return -1e10
        
        D_sq = np.max(pair_dists_sq)
        if D_sq < 1e-16:
            return -1e10
        
        # log_prod = 0.5 * sum(log(pair_dists_sq))
        # - 28 * log(D) = -14 * log(D_sq)
        log_prod = 0.5 * np.sum(np.log(pair_dists_sq))
        log_diam_term = 14.0 * np.log(D_sq)
        
        return log_prod - log_diam_term

    def objective(params):
        """Negative log score for minimization."""
        return -compute_log_score(params)

    def points_to_params(points):
        """Convert 8 points (centered) to 14 parameters."""
        com = np.mean(points, axis=0)
        points = points - com
        xs = points[:, 0]
        ys = points[:, 1]
        return np.append(xs[:7], ys[:7])

    def params_to_points(params):
        """Convert 14 parameters to 8 points."""
        x = params[:7]
        y = params[7:14]
        x7 = -np.sum(x)
        y7 = -np.sum(y)
        xs = np.append(x, x7)
        ys = np.append(y, y7)
        return np.column_stack((xs, ys))

    def normalize_diameter(points):
        """Scale points to have diameter exactly 2."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        D_sq = np.max(dists_sq)
        if D_sq > 0:
            return points * (2.0 / np.sqrt(D_sq))
        return points

    # Generate diverse initial configurations
    def get_octagon(scale=1.0):
        """Regular octagon."""
        angles = np.arange(8) * (np.pi / 4)
        return scale * np.column_stack((np.cos(angles), np.sin(angles)))

    def get_two_squares(r1, r2, angle_offset=0):
        """Two concentric squares."""
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + angle_offset
        angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + angle_offset
        p1 = r1 * np.column_stack((np.cos(angles1), np.sin(angles1)))
        p2 = r2 * np.column_stack((np.cos(angles2), np.sin(angles2)))
        return np.vstack((p1, p2))

    def get_rectangles(w, h):
        """Two rectangles (inner and outer)."""
        # Inner rectangle
        p1 = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        # Outer rectangle, rotated 45 degrees
        w2, h2 = w * 1.5, h * 1.5
        p2 = np.array([[-w2/2, -h2/2], [w2/2, -h2/2], [w2/2, h2/2], [-w2/2, h2/2]])
        # Rotate p2 by 45 degrees
        theta = np.pi/4
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        p2 = p2 @ rot.T
        return np.vstack((p1, p2))

    def get_stretched_octagon(sx, sy):
        """Stretched octagon."""
        angles = np.arange(8) * (np.pi / 4)
        xs = sx * np.cos(angles)
        ys = sy * np.sin(angles)
        return np.column_stack((xs, ys))

    def get_perturbed_octagon(perturbation=0.1, seed=0):
        """Octagon with random perturbations."""
        rng = np.random.default_rng(seed)
        oct = get_octagon()
        oct += rng.normal(0, perturbation, size=oct.shape)
        return oct

    # Create initial configurations
    inits = []
    
    # Regular octagon
    inits.append(get_octagon(1.0))
    
    # Stretched octagons
    inits.append(get_stretched_octagon(1.3, 0.8))
    inits.append(get_stretched_octagon(0.8, 1.3))
    inits.append(get_stretched_octagon(1.5, 0.7))
    inits.append(get_stretched_octagon(0.7, 1.5))
    
    # Two squares with different radii
    inits.append(get_two_squares(0.6, 1.0))
    inits.append(get_two_squares(0.7, 1.0))
    inits.append(get_two_squares(0.8, 1.0))
    inits.append(get_two_squares(0.5, 1.0))
    inits.append(get_two_squares(0.7, 1.2))
    inits.append(get_two_squares(0.8, 1.2))
    
    # Rectangles
    inits.append(get_rectangles(1.0, 0.6))
    inits.append(get_rectangles(0.8, 0.5))
    
    # Perturbed octagons
    for seed in range(5):
        inits.append(get_perturbed_octagon(0.15, seed))
    
    # Random configurations
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(0, 1, (8, 2))
        inits.append(pts)
    
    # Convert all to params
    init_params = [points_to_params(p) for p in inits]
    
    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Optimization loop with time management
    n_inits = len(init_params)
    for i, init in enumerate(init_params):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 5:
            break
        
        # Choose max iterations based on remaining time
        # Rough estimate: each Nelder-Mead iteration takes ~0.1ms for this problem
        # But let's be conservative
        maxiter = min(3000, int(remaining * 50))
        
        if maxiter < 100:
            break
        
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                         options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9,
                                  'adaptive': True})
            
            if res.fun < 1e9:  # Valid result
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = params_to_points(res.x)
                    
        except Exception as e:
            continue
        
        # Also try BFGS for some initializations
        if i % 2 == 0 and remaining > 10:
            try:
                maxiter_bfgs = min(500, int(remaining * 20))
                if maxiter_bfgs > 50:
                    res_bfgs = minimize(objective, init, method='L-BFGS-B',
                                       options={'maxiter': maxiter_bfgs, 'gtol': 1e-8})
                    if res_bfgs.fun < 1e9:
                        score = -res_bfgs.fun
                        if score > best_score:
                            best_score = score
                            best_params = res_bfgs.x.copy()
                            best_points = params_to_points(res_bfgs.x)
            except Exception:
                continue
    
    # If we found a good solution, use it
    if best_params is not None and best_points is not None:
        points = best_points
        points = normalize_diameter(points)
        
        # Final validation: ensure distinct points and valid coordinates
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(8, k=1)
        if np.min(dists[iu]) < 1e-6:
            # Points are too close, fall back
            pass
        elif np.any(np.abs(points) >= 1e6):
            # Coordinates out of bounds, scale down
            points = points * (1e5 / np.max(np.abs(points)))
        else:
            return points.tolist()
    
    # Fallback: return regular octagon
    angles = np.arange(8) * (np.pi / 4)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    points = normalize_diameter(points)
    return points.tolist()
