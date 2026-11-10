# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds
    n = 16

    def compute_score(points):
        """Compute R = (d_min / d_max)^2"""
        if points.shape != (n, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize(points):
        """Center and scale points to unit radius."""
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 1e-15:
            points = points / max_dist
        return points

    def de_objective(x):
        """Objective for differential evolution: minimize -score"""
        points = x.reshape(n, 2)
        # Normalize to keep points within bounds and avoid degenerate scales
        # This helps DE converge faster by standardizing the landscape
        points = normalize(points)
        score = compute_score(points)
        return -score

    def nearest_neighbor_ascent(points, max_rounds=20):
        """
        Heuristic local optimization:
        For each point, move it away from its nearest neighbor if it improves the score.
        """
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                # Find nearest neighbor of point i
                dists = np.linalg.norm(best_points - best_points[i], axis=1)
                dists[i] = np.inf
                nearest_idx = np.argmin(dists)
                
                # Vector from nearest neighbor to point i
                vec = best_points[i] - best_points[nearest_idx]
                norm = np.linalg.norm(vec)
                if norm < 1e-10:
                    continue
                vec /= norm
                
                # Try moving point i away from nearest neighbor
                # Also try moving it towards the center of mass to prevent drift
                center = np.mean(best_points, axis=0)
                vec_to_center = center - best_points[i]
                norm_c = np.linalg.norm(vec_to_center)
                if norm_c > 1e-10:
                    vec_to_center /= norm_c
                else:
                    vec_to_center = np.zeros(2)
                
                # Candidate moves: 
                # 1. Away from nearest neighbor
                # 2. Towards center
                # 3. Random small perturbation
                moves = [
                    vec * 0.05,
                    vec_to_center * 0.05,
                    np.random.randn(2) * 0.02
                ]
                
                for move in moves:
                    new_points = best_points.copy()
                    new_points[i] += move
                    new_points = normalize(new_points)
                    new_score = compute_score(new_points)
                    
                    if new_score > best_score:
                        best_points = new_points
                        best_score = new_score
                        improved = True
                        break # Restart inner loop for this point? Or continue? 
                        # Breaking restarts the direction search for this point, 
                        # but we should probably just accept and move to next point i
                        # Actually, accepting and moving to next i is faster.
                        
            if not improved:
                # If no improvement in a full pass, try smaller steps or break
                # Let's try smaller steps for next round
                # For simplicity, we just break if no improvement after a few rounds
                # But we can also just exit if no improvement
                pass
        return best_points, best_score

    def nelder_mead_polish(points):
        """Polish using Nelder-Mead"""
        if time.time() > deadline - 10:
            return points, compute_score(points)
        
        def nm_obj(x):
            p = x.reshape(n, 2)
            p = normalize(p)
            return -compute_score(p)
        
        try:
            res = minimize(nm_obj, points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or compute_score(res.x.reshape(n,2)) > compute_score(points):
                return normalize(res.x.reshape(n, 2)), compute_score(res.x.reshape(n, 2))
        except Exception:
            pass
        return points, compute_score(points)

    best_points = None
    best_score = -np.inf

    # 1. Run Differential Evolution
    # Bounds: [-2, 2] is safe, normalize handles scale
    bounds = [(-2.0, 2.0)] * (n * 2)
    
    # We allocate ~500 seconds for DE
    de_start = time.time()
    de_budget = 500.0
    
    try:
        # DE is stochastic. Use a good number of iterations.
        # workers=1 is default.
        # tol=1e-8
        res_de = differential_evolution(de_objective, bounds, maxiter=200, 
                                        tol=1e-6, seed=42, polish=False)
        
        if res_de.success or compute_score(res_de.x.reshape(n,2)) > best_score:
            pts = normalize(res_de.x.reshape(n, 2))
            sc = compute_score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts
    except Exception:
        pass

    # 2. Polish the DE result
    if best_points is not None:
        pts, sc = nelder_mead_polish(best_points)
        if sc > best_score:
            best_score = sc
            best_points = pts
        
        pts, sc = nearest_neighbor_ascent(best_points, max_rounds=10)
        if sc > best_score:
            best_score = sc
            best_points = pts

    # 3. Try a few other starting points if time permits
    # Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 4x4 Grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    config_grid = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    config_grid = normalize(config_grid)
    
    # Concentric Rings (8+8)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.6 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = 1.0 * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config_rings = np.vstack([inner_pts, outer_pts])
    config_rings = normalize(config_rings)

    candidates = [config_reg, config_grid, config_rings]
    
    for config in candidates:
        if time.time() > deadline - 30:
            break
        
        # Quick polish from this config
        pts, sc = nearest_neighbor_ascent(config, max_rounds=5)
        if sc > best_score:
            best_score = sc
            best_points = pts
            
        if time.time() > deadline - 20:
            continue
            
        pts, sc = nelder_mead_polish(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts

    # 4. Final Global DE run if time remains? 
    # Or a second DE run with different seed?
    if time.time() < deadline - 100 and best_points is not None:
        # Run DE again with different seed to potentially find a better basin
        try:
            res_de2 = differential_evolution(de_objective, bounds, maxiter=100, 
                                            tol=1e-5, seed=123, polish=False)
            if compute_score(res_de2.x.reshape(n,2)) > best_score:
                pts = normalize(res_de2.x.reshape(n, 2))
                sc = compute_score(pts)
                pts, sc = nelder_mead_polish(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts
        except Exception:
            pass

    # Final polish
    if best_points is not None:
        pts, sc = nearest_neighbor_ascent(best_points, max_rounds=10)
        if sc > best_score:
            best_score = sc
            best_points = pts

    # Fallback
    if best_points is None:
        best_points = config_reg

    return normalize(best_points)

# EVOLVE-BLOCK-END
