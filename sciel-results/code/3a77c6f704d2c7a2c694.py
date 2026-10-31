import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    deadline = time.time() + 115.0
    best_score = 0.0
    best_result = None

    # Aspect ratios to try
    # w + h = 2
    # ar = h/w
    # w = 2 / (1 + ar)
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    def generate_grid_init(w, h, n_circles):
        """
        Generate a grid-based initialization.
        This is robust and guarantees distinct points.
        """
        best_xs, best_ys, best_r = None, None, None
        best_min_dist = 0
        
        for cols in range(1, n_circles + 1):
            rows = int(np.ceil(n_circles / cols))
            if rows == 0: continue
            
            # Spacing
            dx = w / (cols + 1)
            dy = h / (rows + 1)
            
            xs = []
            ys = []
            for r in range(rows):
                for c in range(cols):
                    if len(xs) >= n_circles:
                        break
                    # Center in cell
                    x = (c + 0.5) * dx
                    y = (r + 0.5) * dy
                    xs.append(x)
                    ys.append(y)
                if len(xs) >= n_circles:
                    break
            
            if len(xs) < n_circles:
                continue
                
            xs = np.array(xs[:n_circles])
            ys = np.array(ys[:n_circles])
            
            # Calculate min distance to walls and neighbors
            min_d = np.inf
            for i in range(n_circles):
                min_d = min(min_d, xs[i], w - xs[i], ys[i], h - ys[i])
                for j in range(i+1, n_circles):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_d = min(min_d, d)
            
            # If this grid has better min distance, keep it
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_xs = xs
                best_ys = ys
                # Initial radius is half the min distance to allow growth
                best_r = np.full(n_circles, 0.5 * min_d)
                
        if best_xs is None:
            # Ultimate fallback: single point cluster (bad, but valid if r=0)
            # But we want non-zero. Let's just use a 1xN grid
            xs = np.linspace(0.1, w-0.1, n_circles)
            ys = np.full(n_circles, h/2)
            best_xs = xs
            best_ys = ys
            best_r = np.full(n_circles, 0.05)
            
        return best_xs, best_ys, best_r

    def smooth_penalty(x, y, r, w, h, beta=10.0):
        """
        Smooth penalty for L-BFGS-B.
        Uses softplus to approximate max(0, violation).
        """
        # Wall violations
        # x - r >= 0, w - x - r >= 0, etc.
        viol_walls = np.concatenate([
            r - x,       # x < r
            x + r - w,   # x > w - r
            r - y,       # y < r
            y + r - h    # y > h - r
        ])
        
        # Overlap violations
        # d_ij - r_i - r_j >= 0
        viol_overlaps = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                viol_overlaps.append(r[i] + r[j] - d)
        viol_overlaps = np.array(viol_overlaps)
        
        all_viol = np.concatenate([viol_walls, viol_overlaps])
        
        # Softplus: log(1 + exp(beta * v)) / beta approx max(0, v)
        # To avoid overflow, use np.logaddexp(0, beta * v) / beta
        penalty = np.sum(np.logaddexp(0, beta * all_viol)) / beta
        return penalty

    def objective(vars_, w, h):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        
        # Sum of radii (to maximize, so minimize negative)
        obj = -np.sum(r)
        
        # Penalty
        # We want to strongly penalize violations.
        penalty = smooth_penalty(x, y, r, w, h)
        obj += 500.0 * penalty
        
        return obj

    def validate(x, y, r, w, h):
        if np.any(r < -1e-9): return False
        if np.any(x < r - 1e-7) or np.any(x > w - r + 1e-7): return False
        if np.any(y < r - 1e-7) or np.any(y > h - r + 1e-7): return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-7:
                    return False
        return True

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Generate initial packing
        x0, y0, r0 = generate_grid_init(w, h, N)
        
        # Check heuristic score
        if validate(x0, y0, r0, w, h):
            score_heur = np.sum(r0)
            if score_heur > best_score:
                best_score = score_heur
                best_result = np.column_stack([x0, y0, r0])
        
        # Prepare initial vector for optimizer
        x_init = np.zeros(3*N)
        for i in range(N):
            x_init[3*i] = x0[i]
            x_init[3*i+1] = y0[i]
            x_init[3*i+2] = r0[i]
            
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, min(w, h)/2))
            
        try:
            # L-BFGS-B
            res = minimize(
                objective, 
                x_init, 
                args=(w, h),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            x_opt = res.x[0::3]
            y_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            # Clip radii to be non-negative
            r_opt = np.clip(r_opt, 0, None)
            
            if validate(x_opt, y_opt, r_opt, w, h):
                score_opt = np.sum(r_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = np.column_stack([x_opt, y_opt, r_opt])
                    
                    # Polish
                    if time.time() < deadline - 5:
                        res2 = minimize(
                            objective, 
                            res.x, 
                            args=(w, h),
                            method='L-BFGS-B', 
                            bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-12}
                        )
                        x_pol = res2.x[0::3]
                        y_pol = res2.x[1::3]
                        r_pol = res2.x[2::3]
                        r_pol = np.clip(r_pol, 0, None)
                        
                        if validate(x_pol, y_pol, r_pol, w, h):
                            score_pol = np.sum(r_pol)
                            if score_pol > best_score:
                                best_score = score_pol
                                best_result = np.column_stack([x_pol, y_pol, r_pol])
                                
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        w = 1.0
        h = 1.0
        # Use the robust grid init
        x, y, r = generate_grid_init(w, h, N)
        
        # Final safety check: shrink radii if still invalid
        if not validate(x, y, r, w, h):
             # Shrink until valid
             scale = 1.0
             while not validate(x, y, r*scale, w, h) and scale > 1e-6:
                 scale *= 0.9
             r = r * scale
             
        best_result = np.column_stack([x, y, r])
        best_score = np.sum(r)

    return best_result
