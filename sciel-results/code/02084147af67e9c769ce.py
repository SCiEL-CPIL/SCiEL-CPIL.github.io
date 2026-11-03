"""
Circle packing in unit square using direct geometric optimization.
Optimizes centers and radii simultaneously with SLSQP to maximize sum of radii
under non-overlap and boundary constraints.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def check_validity(x, y, r):
        # Check boundary
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        # Check non-overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def objective(params):
        """Minimize negative sum of radii."""
        r = params[2*N:]
        return -np.sum(r)

    def make_constraints():
        """
        Create constraint functions for SLSQP.
        Constraints are of the form g(params) >= 0.
        1. Boundary: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
        2. Non-overlap: dist_ij - (r_i + r_j) >= 0
        """
        cons = []
        
        # Boundary constraints for x
        # x_i - r_i >= 0
        def c_x_min(p):
            x = p[:N]
            r = p[2*N:]
            return x - r
        cons.append({'type': 'ineq', 'fun': c_x_min})
        
        # 1 - x_i - r_i >= 0
        def c_x_max(p):
            x = p[:N]
            r = p[2*N:]
            return 1.0 - x - r
        cons.append({'type': 'ineq', 'fun': c_x_max})
        
        # Boundary constraints for y
        # y_i - r_i >= 0
        def c_y_min(p):
            y = p[N:2*N]
            r = p[2*N:]
            return y - r
        cons.append({'type': 'ineq', 'fun': c_y_min})
        
        # 1 - y_i - r_i >= 0
        def c_y_max(p):
            y = p[N:2*N]
            r = p[2*N:]
            return 1.0 - y - r
        cons.append({'type': 'ineq', 'fun': c_y_max})
        
        # Non-overlap constraints
        # dist_ij - (r_i + r_j) >= 0
        # We can vectorize this by returning a flat array of all pair constraints
        # However, SLSQP expects 'fun' to return an array.
        # Number of pairs is N*(N-1)/2.
        
        def c_non_overlap(p):
            x = p[:N]
            y = p[N:2*N]
            r = p[2*N:]
            
            # Compute all pairwise distances and required distances
            # Using broadcasting
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            req_dist = r[:, None] + r[None, :]
            
            # We want dist_ij - req_dist_ij >= 0 for i != j
            # Create a mask for upper triangle (excluding diagonal)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Extract values
            vals = dist[mask] - req_dist[mask]
            return vals
        
        cons.append({'type': 'ineq', 'fun': c_non_overlap})
        
        return cons

    def make_initial_guess(seed):
        """
        Generate a feasible initial configuration.
        We use a grid layout with small radii to ensure feasibility,
        then let the optimizer expand them.
        """
        rng = np.random.RandomState(seed)
        
        # Grid layout
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        
        # Add small random perturbation
        x += rng.uniform(-0.02, 0.02, N)
        y += rng.uniform(-0.02, 0.02, N)
        
        # Clip to [0.01, 0.99]
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        # Calculate initial radii based on local geometry to ensure feasibility
        # r_i = min(0.5 * min_dist_to_neighbor, boundary_dist) * 0.8 (safety factor)
        r = np.zeros(N)
        for i in range(N):
            # Boundary
            rb = np.min([x[i], 1-x[i], y[i], 1-y[i]])
            
            # Neighbors
            min_d = np.inf
            for j in range(N):
                if i == j: continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < min_d:
                    min_d = d
            
            # Heuristic radius
            r_i = np.min([rb, 0.5 * min_d])
            # Safety factor to ensure strict feasibility
            r[i] = r_i * 0.7
            
        # Ensure r > 0
        r = np.maximum(r, 0.001)
        
        params = np.concatenate([x, y, r])
        
        # Check if feasible, if not, shrink radii
        if not check_validity(x, y, r):
            for scale in [0.5, 0.3, 0.1]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    r = r_test
                    break
        
        return params

    def run_optimization(seed):
        if time.time() > DEADLINE:
            return None
            
        x0, y0, r0 = make_initial_guess(seed)
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds:
        # x, y in [0.01, 0.99] (radii will be limited by constraints)
        # r in [0.001, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99)) # x
        for i in range(N):
            bounds.append((0.01, 0.99)) # y
        for i in range(N):
            bounds.append((0.001, 0.5)) # r
            
        constraints = make_constraints()
        
        options = {'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        
        try:
            res = minimize(objective, params0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options=options)
            
            if res.success:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                # Final validity check
                if check_validity(x_opt, y_opt, r_opt):
                    val = np.sum(r_opt)
                    return x_opt, y_opt, r_opt, val
                else:
                    # Try to repair
                    for scale in [0.99, 0.95, 0.90]:
                        r_test = r_opt * scale
                        if check_validity(x_opt, y_opt, r_test):
                            return x_opt, y_opt, r_test, np.sum(r_test)
            return None
        except Exception:
            return None

    # Initial best: Grid configuration
    best_params = make_initial_guess(0)
    best_x = best_params[:N]
    best_y = best_params[N:2*N]
    best_r = best_params[2*N:]
    best_val = np.sum(best_r)

    # Multi-start loop
    num_starts = 15
    for i in range(num_starts):
        if time.time() > DEADLINE:
            break

        res = run_optimization(i)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 10):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            # Fallback to initial grid
            fallback = make_initial_guess(0)
            best_x = fallback[:N]
            best_y = fallback[N:2*N]
            best_r = fallback[2*N:]

    return np.column_stack((best_x, best_y, best_r))
