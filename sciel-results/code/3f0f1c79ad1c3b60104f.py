import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 10.0

    def get_cost(pts):
        """Calculate (d_max / d_min)^2. Lower is better."""
        d = pdist(pts)
        if len(d) == 0:
            return np.inf
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return np.inf
        return (d_max / d_min) ** 2

    def hexagonal_lattice_init(n_points=16):
        """Generate a hexagonal lattice and select the n_points closest to the centroid."""
        # Generate a large hexagonal lattice
        R = 10
        points = []
        # Hex lattice basis
        a = 1.0
        b = np.sqrt(3) / 2
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i * a + j * 0.5 * a
                y = j * b
                points.append([x, y])
        
        points = np.array(points)
        
        # Find the subset of n_points that minimizes the diameter (max distance)
        # Heuristic: pick points closest to the mean of the whole set? No, that might be sparse.
        # Better: pick a window of points.
        # Let's find the centroid of all points, then pick the n_points closest to that centroid.
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        idx = np.argsort(dists)
        selected = points[idx[:n_points]]
        
        # Center the selected points
        selected = selected - np.mean(selected, axis=0)
        return selected

    def grid_init(n_points=16):
        """Generate a square grid."""
        side = int(np.ceil(np.sqrt(n_points)))
        # 4x4 grid
        g = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
        g = g - np.mean(g, axis=0)
        # If we have more points than needed, trim
        if len(g) > n_points:
            centroid = np.mean(g, axis=0)
            dists = np.linalg.norm(g - centroid, axis=1)
            idx = np.argsort(dists)
            g = g[idx[:n_points]]
        return g

    def random_init(n_points=16):
        return np.random.randn(n_points, 2)

    # Generate initial candidates
    candidates = [
        hexagonal_lattice_init(16),
        grid_init(16),
        random_init(16)
    ]
    
    # Evaluate initial candidates
    best_pts = None
    best_cost = np.inf
    for c in candidates:
        c = c - np.mean(c, axis=0) # Ensure centered
        cost = get_cost(c)
        if cost < best_cost:
            best_cost = cost
            best_pts = c.copy()

    # Simulated Annealing Refinement
    # We will run SA on the best candidate found so far.
    # We can also run SA on multiple candidates if time permits.
    
    def simulated_annealing(initial_pts, time_budget):
        pts = initial_pts.copy()
        current_cost = get_cost(pts)
        best_pts_sa = pts.copy()
        best_cost_sa = current_cost
        
        # SA Parameters
        T = 0.1  # Initial temperature (scale of perturbation)
        cooling_rate = 0.999
        min_T = 1e-6
        max_iter = 1000000
        no_improve_count = 0
        max_no_improve = 10000
        
        t_start = time.time()
        
        for i in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            
            # Perturb a random point
            idx = np.random.randint(0, len(pts))
            noise = np.random.randn(2) * T
            new_pts = pts.copy()
            new_pts[idx] += noise
            
            new_cost = get_cost(new_pts)
            
            # Acceptance criterion
            if new_cost < current_cost:
                pts = new_pts
                current_cost = new_cost
                no_improve_count = 0
                
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pts_sa = pts.copy()
            else:
                # Metropolis criterion
                delta = new_cost - current_cost
                if delta < 0:
                    accept = True
                else:
                    # If T is very small, exp(-delta/T) will be 0
                    if T > min_T:
                        prob = np.exp(-delta / T)
                        accept = np.random.rand() < prob
                    else:
                        accept = False
                
                if accept:
                    pts = new_pts
                    current_cost = new_cost
                else:
                    no_improve_count += 1
                    
            # Cool down
            T *= cooling_rate
            
            # Restart if stuck
            if no_improve_count > max_no_improve:
                # Reset to best found and re-heat
                pts = best_pts_sa.copy()
                current_cost = best_cost_sa
                T = 0.05
                no_improve_count = 0
                max_no_improve = 5000 # Stricter next time

        return best_pts_sa, best_cost_sa

    # Allocate time for SA
    # We have ~600s total. Let's spend ~500s on SA and ~50s on final polishing.
    time_elapsed = time.time() - start_time
    sa_time_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 60.0 # Leave 60s for polishing/fallback
    if sa_time_budget > 10:
        # Run SA on the best initial candidate
        best_pts_sa, best_cost_sa = simulated_annealing(best_pts, sa_time_budget)
        if best_cost_sa < best_cost:
            best_pts = best_pts_sa
            best_cost = best_cost_sa
            
        # If time permits, try SA on other candidates? 
        # Usually one good SA run is better than many short ones.
        # But let's try a second short SA on the second best candidate if we have time.
        time_elapsed = time.time() - start_time
        remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 60.0
        if remaining > 30:
             # Find second best candidate
             c2 = grid_init(16) if best_pts is not candidates[0] else candidates[0]
             c2 = c2 - np.mean(c2, axis=0)
             best_pts_sa2, best_cost_sa2 = simulated_annealing(c2, remaining)
             if best_cost_sa2 < best_cost:
                 best_pts = best_pts_sa2
                 best_cost = best_cost_sa2

    # Final Polishing with Nelder-Mead
    time_elapsed = time.time() - start_time
    polish_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    if polish_budget > 5:
        def objective(params):
            pts = params.reshape(16, 2)
            d = pdist(pts)
            if len(d) == 0:
                return 1e10
            d_min = np.min(d)
            d_max = np.max(d)
            if d_min <= 1e-12 or d_max <= 1e-12:
                return 1e10
            return (d_max / d_min) ** 2

        initial_params = best_pts.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                res_cost = get_cost(res_pts)
                if res_cost < best_cost:
                    best_pts = res_pts
                    best_cost = res_cost
        except Exception:
            pass

    # Final check and return
    final_pts = best_pts
    if final_pts is None:
        final_pts = np.random.randn(16, 2)
        
    if not np.all(np.isfinite(final_pts)):
        final_pts = grid_init(16)

    return final_pts
