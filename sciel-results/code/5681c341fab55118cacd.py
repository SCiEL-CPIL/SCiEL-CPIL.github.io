import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    deadline = time.time() + 140  # Safety margin
    
    def smooth_max(x, beta=100.0):
        """Smooth approximation of max(0, x)"""
        # Using softplus: log(1 + exp(beta*x)) / beta
        # For large beta, approximates max(0, x)
        # For stability, handle large values
        arg = beta * x
        # np.logaddexp(0, arg) is log(1 + exp(arg))
        return np.logaddexp(0.0, arg) / beta

    def evaluate(w, centers, radii):
        """Check validity and return sum of radii if valid."""
        h = 2.0 - w
        if h <= 0 or w <= 0:
            return -np.inf
        # Check bounds
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if r < 0:
                return -np.inf
            if x < r - 1e-9 or x > w - r + 1e-9 or y < r - 1e-9 or y > h - r + 1e-9:
                return -np.inf
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist2 = dx*dx + dy*dy
                if dist2 < (radii[i]+radii[j])**2 - 1e-9:
                    return -np.inf
        return np.sum(radii)

    def objective(params, w, beta, penalty_weight):
        """
        Objective function for L-BFGS-B.
        Minimizes -sum(radii) + penalty.
        """
        h = 2.0 - w
        centers = params[:2*N].reshape(N, 2)
        radii = params[2*N:]
        
        # Base objective: maximize sum of radii
        obj = -np.sum(radii)
        
        penalty = 0.0
        
        # Boundary constraints:
        # x >= r, x <= w - r, y >= r, y <= h - r
        # Violation is positive if constraint is broken.
        # v1 = r - x
        # v2 = x - (w - r)
        # v3 = r - y
        # v4 = y - (h - r)
        
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            
            v1 = r - x
            v2 = x - (w - r)
            v3 = r - y
            v4 = y - (h - r)
            
            penalty += smooth_max(v1, beta)
            penalty += smooth_max(v2, beta)
            penalty += smooth_max(v3, beta)
            penalty += smooth_max(v4, beta)
            
            # Radius non-negative
            v5 = -r
            penalty += smooth_max(v5, beta)
            
        # Overlap constraints:
        # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
        # Let d = sqrt(dx^2 + dy^2). Violation v = (ri + rj) - d.
        # To avoid sqrt derivative issues near 0, we can use squared distance?
        # No, smooth_max needs the linear violation.
        # We can approximate d with a smooth function or just use sqrt with a small epsilon.
        
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist2 = dx*dx + dy*dy
                d = np.sqrt(dist2 + 1e-12)
                min_dist = radii[i] + radii[j]
                v = min_dist - d
                penalty += smooth_max(v, beta)
        
        return obj + penalty_weight * penalty

    def make_initial(w_init, mode='grid'):
        h_init = 2.0 - w_init
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        if mode == 'grid':
            cols = 5
            rows = 5
            x_positions = np.linspace(0.5*w_init/cols, 0.5*(w_init - w_init/cols), cols)
            y_positions = np.linspace(0.5*h_init/rows, 0.5*(h_init - h_init/rows), rows)
            idx = 0
            for i in range(cols):
                for j in range(rows):
                    if idx >= N: break
                    centers[idx, 0] = x_positions[i]
                    centers[idx, 1] = y_positions[j]
                    cell_w = w_init / cols
                    cell_h = h_init / rows
                    radii[idx] = 0.4 * min(cell_w, cell_h)
                    idx += 1
        elif mode == 'hex':
            # Hexagonal packing
            # Approximate radius
            r0 = min(w_init, h_init) / (2 * (N**0.5)) * 0.9
            dx = 2 * r0
            dy = np.sqrt(3) * r0
            idx = 0
            y = r0
            row = 0
            while idx < N and y < h_init - r0:
                x = r0 + (row % 2) * (dx / 2)
                while x < w_init - r0 and idx < N:
                    centers[idx, 0] = x
                    centers[idx, 1] = y
                    radii[idx] = r0
                    idx += 1
                    x += dx
                y += dy
                row += 1
            # If we didn't fill all, scale up radii slightly or just leave them
        elif mode == 'random':
            centers[:, 0] = np.random.uniform(0.1, w_init - 0.1, N)
            centers[:, 1] = np.random.uniform(0.1, h_init - 0.1, N)
            radii = np.random.uniform(0.05, 0.1, N)
            
        return centers, radii

    best_score = -np.inf
    best_result = None
    
    # Try different initial aspect ratios and modes
    w_inits = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3]
    modes = ['grid', 'hex', 'random']
    
    # Parameters for optimization
    penalty_weight = 100.0
    beta_start = 10.0
    beta_end = 100.0
    
    for w_init in w_inits:
        if time.time() > deadline:
            break
        h_init = 2.0 - w_init
        if h_init <= 0.1:
            continue
            
        for mode in modes:
            if time.time() > deadline:
                break
                
            centers, radii = make_initial(w_init, mode)
            params0 = np.concatenate([[w_init], centers.flatten(), radii])
            
            # Bounds
            # w: (0.1, 1.9)
            # x_i: (0, 2.0)
            # y_i: (0, 2.0)
            # r_i: (0.001, 1.0)
            bounds = [(0.1, 1.9)] + [(0, 2.0)]*(2*N) + [(0.001, 1.0)]*N
            
            # Successive beta increase
            current_params = params0.copy()
            
            for beta in [beta_start, 50.0, beta_end]:
                if time.time() > deadline:
                    break
                
                def obj_wrapper(p):
                    w = p[0]
                    rest = p[1:]
                    return objective(rest, w, beta, penalty_weight)
                
                result = minimize(obj_wrapper, current_params, method='L-BFGS-B', 
                                 bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.success:
                    current_params = result.x
                    w_opt = current_params[0]
                    c_opt = current_params[1:1+2*N].reshape(N, 2)
                    r_opt = current_params[1+2*N:]
                    
                    # Check validity and score
                    score = evaluate(w_opt, c_opt, r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = (w_opt, c_opt, r_opt)
    
    # Final Polish with very tight beta and high penalty
    if best_result is not None and time.time() < deadline:
        w_opt, c_opt, r_opt = best_result
        h_opt = 2.0 - w_opt
        
        params0 = np.concatenate([[w_opt], c_opt.flatten(), r_opt])
        bounds = [(0.1, 1.9)] + [(0, 2.0)]*(2*N) + [(0.001, 1.0)]*N
        
        for beta in [1000.0, 5000.0]:
            if time.time() > deadline:
                break
            
            def obj_wrapper_polish(p):
                w = p[0]
                rest = p[1:]
                return objective(rest, w, beta, 1000.0)
            
            result = minimize(obj_wrapper_polish, params0, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.success:
                w_opt = result.x[0]
                c_opt = result.x[1:1+2*N].reshape(N, 2)
                r_opt = result.x[1+2*N:]
                score = evaluate(w_opt, c_opt, r_opt)
                if score > best_score:
                    best_score = score
                    best_result = (w_opt, c_opt, r_opt)

    if best_result is None:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i+0.5)*w/side
                y = (j+0.5)*h/side
                r = 0.5*min(w,h)/side - 1e-4
                cells.append([x, y, r])
        cells = cells[:N]
        return np.array(cells)
    
    w_opt, c_opt, r_opt = best_result
    result = np.column_stack([c_opt, r_opt])
    return result
