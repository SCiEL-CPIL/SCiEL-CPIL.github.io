import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin before 120s limit

    def compute_inradii(x, y):
        """
        Given center positions, compute the maximum inradius for each circle
        using Voronoi diagram. The inradius of circle i is the minimum distance
        from center i to any Voronoi edge or boundary.
        """
        points = np.column_stack((x, y))
        r = np.zeros(N)
        
        # Add boundary points to constrain Voronoi to [0,1]^2
        # Use a large bounding box approach with clipped Voronoi
        try:
            vor = Voronoi(points)
        except Exception:
            # Fallback: use pairwise distances
            for i in range(N):
                dists = np.sqrt((x[i] - x)**2 + (y[i] - y)**2)
                dists[i] = np.inf
                # Distance to boundaries
                dists[i] = min(x[i], 1 - x[i], y[i], 1 - y[i])
                r[i] = min(dists[i] / 2, min(x[i], 1 - x[i], y[i], 1 - y[i]))
            return r
        
        # For each site, find the minimum distance to any Voronoi vertex or edge
        for i in range(N):
            min_dist = min(x[i], 1 - x[i], y[i], 1 - y[i])  # Distance to boundary
            
            # Check Voronoi vertices
            if i in vor.point_region:
                region = vor.regions[vor.point_region[i]]
                if len(region) > 0:
                    for v_idx in region:
                        if v_idx != -1:
                            vx, vy = vor.vertices[v_idx]
                            d = np.sqrt((x[i] - vx)**2 + (y[i] - vy)**2)
                            min_dist = min(min_dist, d)
            
            r[i] = min_dist
        
        return r

    def voronoi_objective(params):
        """
        Objective: minimize negative sum of inradii.
        params: 2N array of x and y coordinates.
        """
        x = params[0::2]
        y = params[1::2]
        
        # Clamp to valid range (with small margin)
        margin = 0.01
        x = np.clip(x, margin, 1 - margin)
        y = np.clip(y, margin, 1 - margin)
        
        r = compute_inradii(x, y)
        return -np.sum(r)

    def voronoi_gradient(params):
        """
        Numerical gradient of the Voronoi objective.
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            grad[i] = (voronoi_objective(params_plus) - voronoi_objective(params_minus)) / (2 * eps)
        return grad

    def voronoi_polish(x, y, max_iter=30, time_limit=None):
        """
        Optimize center positions using gradient ascent on sum of inradii.
        """
        params = np.column_stack((x, y)).flatten()
        
        def objective(p):
            if time_limit is not None and time.time() > time_limit:
                return 1e10
            return voronoi_objective(p)
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                          jac=voronoi_gradient,
                          options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
            x_opt = res.x[0::2]
            y_opt = res.x[1::2]
        except Exception:
            x_opt = x.copy()
            y_opt = y.copy()
        
        # Clamp
        margin = 0.005
        x_opt = np.clip(x_opt, margin, 1 - margin)
        y_opt = np.clip(y_opt, margin, 1 - margin)
        
        r_opt = compute_inradii(x_opt, y_opt)
        return x_opt, y_opt, r_opt

    def check_validity(x, y, r, tol=1e-7):
        """Check if circles are valid (non-overlapping, within bounds)."""
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True

    def make_grid_initial(rng):
        """Create a grid-based initial guess for centers."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
        # Add small random perturbation
        x += rng.normal(0, 0.02, N)
        y += rng.normal(0, 0.02, N)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def make_random_initial(rng):
        """Create random initial guess for centers."""
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y

    def voronoi_sa(x, y, temp, rng, n_steps=100):
        """
        Simulated annealing on center positions, using Voronoi inradii as fitness.
        """
        best_score = -np.inf
        best_x, best_y = x.copy(), y.copy()
        
        current_score = -np.sum(compute_inradii(x, y))
        
        for step in range(n_steps):
            if time.time() > deadline - 10:
                break
            
            # Perturb random subset
            n_perturb = max(1, N // 5)
            idx = rng.choice(N, size=n_perturb, replace=False)
            
            step_size = 0.03 * temp
            x_new = x.copy()
            y_new = y.copy()
            x_new[idx] += rng.normal(0, step_size, size=len(idx))
            y_new[idx] += rng.normal(0, step_size, size=len(idx))
            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)
            
            new_score = -np.sum(compute_inradii(x_new, y_new))
            
            # Metropolis acceptance
            if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / max(temp, 1e-10)):
                x, y = x_new, y_new
                current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_x, best_y = x.copy(), y.copy()
            
            temp *= 0.997
        
        return best_x, best_y, -best_score

    rng = np.random.RandomState(42)
    
    best_x, best_y, best_r = None, None, None
    best_sum_r = -np.inf
    
    # Phase 1: Grid initialization + Voronoi SA + polish
    x, y = make_grid_initial(rng)
    x, y, score = voronoi_sa(x, y, temp=1.0, rng=rng, n_steps=2000)
    if time.time() < deadline - 15:
        x, y, r = voronoi_polish(x, y, max_iter=20, time_limit=deadline - 10)
        if check_validity(x, y, r):
            sum_r = np.sum(r)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 2: Multiple random restarts
    remaining = deadline - time.time()
    n_restarts = max(1, int(remaining / 12))
    
    for restart in range(n_restarts):
        if time.time() > deadline - 8:
            break
        
        x_init, y_init = make_random_initial(rng)
        
        # Quick SA
        x_s, y_s, _ = voronoi_sa(x_init, y_init, temp=0.5, rng=rng, n_steps=300)
        
        # Polish
        if time.time() < deadline - 10:
            x_p, y_p, r_p = voronoi_polish(x_s, y_s, max_iter=15, time_limit=deadline - 8)
            if check_validity(x_p, y_p, r_p):
                sum_r = np.sum(r_p)
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()
    
    # Phase 3: Local search - try moving individual circles
    if best_x is not None and time.time() < deadline - 5:
        x_ls, y_ls, r_ls = best_x.copy(), best_y.copy(), best_r.copy()
        for iteration in range(50):
            if time.time() > deadline - 4:
                break
            i = rng.randint(N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            x_try = x_ls.copy()
            y_try = y_ls.copy()
            x_try[i] = np.clip(x_try[i] + dx, 0.02, 0.98)
            y_try[i] = np.clip(y_try[i] + dy, 0.02, 0.98)
            r_try = compute_inradii(x_try, y_try)
            if check_validity(x_try, y_try, r_try):
                sum_r = np.sum(r_try)
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_x, best_y, best_r = x_try.copy(), y_try.copy(), r_try.copy()
                    x_ls, y_ls, r_ls = x_try.copy(), y_try.copy(), r_try.copy()
    
    # Fallback if no valid result
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.8
        return np.column_stack((x, y, r))
    
    # Final repair: ensure validity
    x, y, r = best_x, best_y, best_r
    for _ in range(30):
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        if np.max(viol) < 1e-8:
            break
        idx_i, idx_j = np.unravel_index(np.argmax(viol), viol.shape)
        reduction = viol[idx_i, idx_j] / 2 + 1e-7
        r[idx_i] = max(r[idx_i] - reduction, 1e-4)
        r[idx_j] = max(r[idx_j] - reduction, 1e-4)
    
    # Final clamp
    r = np.maximum(r, 1e-4)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.column_stack((x, y, r))
