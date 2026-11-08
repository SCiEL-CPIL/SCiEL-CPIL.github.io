import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    """
    New Approach: Iterative Relaxation with Constraint Projection
    
    Instead of SA + L-BFGS, we use:
    1. Start with a good initial configuration (grid-based)
    2. Iteratively:
       a. Scale radii to maximize sum while maintaining feasibility
       b. Move circles to reduce overlaps (repulsion forces)
       c. Project circles back into the rectangle
       d. Optimize aspect ratio w via 1D search
    3. Use Nelder-Mead for final polishing of positions and radii
    """
    
    def compute_radius_sum(x, y, r, w):
        """Sum of radii"""
        return np.sum(r)
    
    def check_validity(x, y, r, w, tol=1e-6):
        """Check if configuration is valid"""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        if np.any(r <= 0):
            return False
        
        # Boundary check
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True
    
    def relax_overlaps(x, y, r, w, h, n_iter=100, step_scale=0.5):
        """
        Iteratively move circles to reduce overlaps using repulsion forces.
        Returns improved x, y, r.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(n_iter):
            # Compute overlap forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            overlap_found = False
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        overlap_found = True
                        # Repulsion force proportional to overlap
                        overlap = min_dist - dist
                        force_mag = overlap * 10.0
                        fx[i] += (dx / dist) * force_mag
                        fy[i] += (dy / dist) * force_mag
                        fx[j] -= (dx / dist) * force_mag
                        fy[j] -= (dy / dist) * force_mag
                        # Slightly shrink radii to help resolve
                        fr[i] -= overlap * 0.1
                        fr[j] -= overlap * 0.1
            
            if not overlap_found:
                break
            
            # Apply forces with damping
            step = step_scale / (1.0 + it * 0.1)
            x += fx * step
            y += fy * step
            r += fr * step
            
            # Enforce minimum radius
            r = np.maximum(r, 0.01)
            
            # Project back into rectangle
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        
        return x, y, r
    
    def scale_radii(x, y, r, w, h, target_ratio=0.95):
        """
        Scale radii uniformly to approach the maximum feasible size.
        Uses binary search on scale factor.
        """
        # Try scaling up
        def is_feasible(scale):
            r_test = r * scale
            for i in range(N):
                if r_test[i] > w/2 or r_test[i] > h/2:
                    return False
                if x[i] < r_test[i] or x[i] > w - r_test[i]:
                    return False
                if y[i] < r_test[i] or y[i] > h - r_test[i]:
                    return False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_test[i] + r_test[j]:
                        return False
            return True
        
        # Binary search for max scale
        lo, hi = 0.5, 2.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if is_feasible(mid):
                lo = mid
            else:
                hi = mid
        
        return r * lo
    
    def optimize_aspect_ratio(x, y, r, w_init, n_trials=20):
        """
        Try different aspect ratios and find the best one.
        """
        best_score = 0
        best_w = w_init
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Try a range of aspect ratios
        for w in np.linspace(0.6, 1.4, n_trials):
            h = 2.0 - w
            
            # Rescale positions to fit new rectangle
            x_new = x * (w / w_init)
            y_new = y * (h / (2.0 - w_init))
            
            # Relax overlaps
            x_rel, y_rel, r_rel = relax_overlaps(x_new, y_new, r.copy(), w, h, n_iter=50)
            
            # Scale radii
            r_scaled = scale_radii(x_rel, y_rel, r_rel, w, h)
            
            # Check validity
            if check_validity(x_rel, y_rel, r_scaled, w, tol=1e-5):
                score = np.sum(r_scaled)
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_x = x_rel.copy()
                    best_y = y_rel.copy()
                    best_r = r_scaled.copy()
        
        return best_w, best_x, best_y, best_r
    
    def polish_with_nelder_mead(x, y, r, w, h):
        """
        Use Nelder-Mead to polish the configuration.
        """
        def objective(params):
            w_p = params[0]
            h_p = 2.0 - w_p
            if w_p <= 0.1 or h_p <= 0.1:
                return 1e10
            
            x_p = params[1:1+N]
            y_p = params[1+N:1+2*N]
            r_p = params[1+2*N:1+3*N]
            
            # Penalty for invalidity
            penalty = 0.0
            for i in range(N):
                if r_p[i] <= 0:
                    penalty += 100.0
                if x_p[i] < r_p[i]:
                    penalty += (r_p[i] - x_p[i])**2 * 50.0
                elif x_p[i] > w_p - r_p[i]:
                    penalty += (x_p[i] - (w_p - r_p[i]))**2 * 50.0
                if y_p[i] < r_p[i]:
                    penalty += (r_p[i] - y_p[i])**2 * 50.0
                elif y_p[i] > h_p - r_p[i]:
                    penalty += (y_p[i] - (h_p - r_p[i]))**2 * 50.0
                if r_p[i] > w_p/2:
                    penalty += (r_p[i] - w_p/2)**2 * 50.0
                if r_p[i] > h_p/2:
                    penalty += (r_p[i] - h_p/2)**2 * 50.0
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_p[i] - x_p[j]
                    dy = y_p[i] - y_p[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r_p[i] + r_p[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2 * 50.0
            
            return -np.sum(r_p) + penalty
        
        # Initial params
        params0 = np.concatenate([[w], x, y, r])
        
        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < 0:  # Valid solution
                w_res = res.x[0]
                x_res = res.x[1:1+N]
                y_res = res.x[1+N:1+2*N]
                r_res = res.x[1+2*N:1+3*N]
                
                if check_validity(x_res, y_res, r_res, w_res, tol=1e-5):
                    return x_res, y_res, r_res, w_res
        except:
            pass
        
        return x, y, r, w
    
    start_time = time.time()
    time_limit = 110.0
    
    best_score = 0.0
    best_x, best_y, best_r, best_w = None, None, None, None
    
    # Try multiple initial configurations
    for trial in range(3):
        if time.time() - start_time > time_limit - 30:
            break
        
        np.random.seed(100 + trial)
        
        # Initial aspect ratio
        w_init = np.random.uniform(0.7, 1.3)
        h_init = 2.0 - w_init
        
        # Initialize in a grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            col = i % side
            row = i // side
            x[i] = (col + 0.5) * (w_init / side)
            y[i] = (row + 0.5) * (h_init / side)
            r[i] = min(w_init, h_init) / (side * 2) * 0.9
        
        # Step 1: Relax overlaps
        x, y, r = relax_overlaps(x, y, r, w_init, h_init, n_iter=200)
        
        # Step 2: Scale radii
        r = scale_radii(x, y, r, w_init, h_init)
        
        # Step 3: Optimize aspect ratio
        w_opt, x_opt, y_opt, r_opt = optimize_aspect_ratio(x, y, r, w_init)
        
        # Step 4: Polish with Nelder-Mead
        h_opt = 2.0 - w_opt
        x_pol, y_pol, r_pol, w_pol = polish_with_nelder_mead(x_opt, y_opt, r_opt, w_opt, h_opt)
        
        # Check and update best
        if check_validity(x_pol, y_pol, r_pol, w_pol, tol=1e-5):
            score = np.sum(r_pol)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r, best_w = x_pol.copy(), y_pol.copy(), r_pol.copy(), w_pol
        
        # Also try the pre-polish solution
        if check_validity(x_opt, y_opt, r_opt, w_opt, tol=1e-5):
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r, best_w = x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt
    
    if best_x is None:
        # Fallback: simple grid in 1x1
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        result = []
        for i in range(N):
            col = i % side
            row = i // side
            xi = (col + 0.5) / side
            yi = (row + 0.5) / side
            ri = 0.5 / side - 1e-4
            result.append([xi, yi, ri])
        return np.array(result)
    
    # Extract result
    result = []
    for i in range(N):
        result.append([best_x[i], best_y[i], best_r[i]])
    
    return np.array(result)
