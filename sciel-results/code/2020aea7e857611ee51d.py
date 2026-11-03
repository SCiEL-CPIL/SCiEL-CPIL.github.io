import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s buffer

def construct():
    def time_remaining():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    def place_circle_greedy(placed, W, H):
        """
        Places a single circle given a list of placed circles.
        Uses SLSQP to maximize radius r for the new circle (x,y,r).
        Returns (x, y, r) or None if no valid placement found.
        """
        if not placed:
            # First circle: center it, max radius is min(W,H)/2
            cx, cy = W/2.0, H/2.0
            r = min(W, H) / 2.0
            return cx, cy, r

        # Objective: minimize -r
        def obj(v):
            x, y, r = v
            return -r

        # Constraints:
        # 1. x - r >= 0
        # 2. x + r <= W
        # 3. y - r >= 0
        # 4. y + r <= H
        # 5. r > 0
        # 6. For each placed circle (px, py, pr): (x-px)^2 + (y-py)^2 >= (r+pr)^2
        
        constraints = [
            {'type': 'ineq', 'fun': lambda v: v[0] - v[2]}, # x >= r
            {'type': 'ineq', 'fun': lambda v: W - v[0] - v[2]}, # x + r <= W
            {'type': 'ineq', 'fun': lambda v: v[1] - v[2]}, # y >= r
            {'type': 'ineq', 'fun': lambda v: H - v[1] - v[2]}, # y + r <= H
            {'type': 'ineq', 'fun': lambda v: v[2]} # r > 0
        ]
        
        for px, py, pr in placed:
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, px=px, py=py, pr=pr: (v[0]-px)**2 + (v[1]-py)**2 - (v[2]+pr)**2
            })

        # Initial guess: try center, small radius
        # Or try several starting points
        best_r = -1
        best_v = None
        
        # Start from center
        x0, y0, r0 = W/2.0, H/2.0, 0.05
        try:
            res = minimize(obj, [x0, y0, r0], method='SLSQP', constraints=constraints, 
                           options={'maxiter': 50, 'ftol': 1e-6})
            if res.success and res.fun < 0:
                r_val = -res.fun
                if r_val > best_r:
                    best_r = r_val
                    best_v = res.x
        except Exception:
            pass
            
        # If center didn't work well, try corners/edges
        starts = [
            [0.1, 0.1, 0.05],
            [W-0.1, 0.1, 0.05],
            [0.1, H-0.1, 0.05],
            [W-0.1, H-0.1, 0.05],
            [W/2.0, 0.1, 0.05],
            [W/2.0, H-0.1, 0.05]
        ]
        
        for s in starts:
            # Ensure start is feasible for boundaries roughly
            if s[0] < s[2] or s[0] + s[2] > W or s[1] < s[2] or s[1] + s[2] > H:
                continue
            try:
                res = minimize(obj, s, method='SLSQP', constraints=constraints, 
                               options={'maxiter': 50, 'ftol': 1e-6})
                if res.success and res.fun < 0:
                    r_val = -res.fun
                    if r_val > best_r:
                        best_r = r_val
                        best_v = res.x
            except Exception:
                pass

        if best_v is not None:
            return best_v[0], best_v[1], best_v[2]
        return None

    def refine_configuration(x, y, r, W, H, iterations=100):
        """
        Local search to expand radii and adjust positions.
        """
        current_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_score = current_score
        
        step = 0.01
        for _ in range(iterations):
            if time_remaining() < 5:
                break
            improved = False
            # Try to expand each radius
            for i in range(N):
                # Check if we can expand r[i]
                new_r = r[i] + step
                if new_r <= 0: continue
                if x[i] - new_r < 0 or x[i] + new_r > W: continue
                if y[i] - new_r < 0 or y[i] + new_r > H: continue
                
                valid = True
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    rad_sum = new_r + r[j]
                    if dist_sq < rad_sum*rad_sum - 1e-8:
                        valid = False
                        break
                
                if valid:
                    r[i] = new_r
                    improved = True
            
            # If no radius could be expanded, try perturbing positions
            if not improved:
                # Random perturbation
                mask = np.random.rand(N) > 0.5
                x_new = x.copy()
                y_new = y.copy()
                x_new[mask] += np.random.normal(0, 0.05, np.sum(mask))
                y_new[mask] += np.random.normal(0, 0.05, np.sum(mask))
                
                # Clip to bounds (approximate, keeping radii same for now)
                # We need to ensure they don't go out of bounds
                for i in range(N):
                    x_new[i] = np.clip(x_new[i], r[i], W - r[i])
                    y_new[i] = np.clip(y_new[i], r[i], H - r[i])
                
                # Check validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_new[i] - x_new[j]
                        dy = y_new[i] - y_new[j]
                        dist_sq = dx*dx + dy*dy
                        rad_sum = r[i] + r[j]
                        if dist_sq < rad_sum*rad_sum - 1e-8:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    x, y = x_new, y_new
                    improved = True
            
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
                    
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        return best_x, best_y, best_r, best_score

    best_overall_score = -np.inf
    best_result = None
    
    # Search over aspect ratios W. H = 2 - W.
    # W ranges from 0.5 to 1.5
    W_values = np.linspace(0.6, 1.4, 9)
    
    for W in W_values:
        if time_remaining() < 10:
            break
        H = 2.0 - W
        
        placed = []
        x_list = []
        y_list = []
        r_list = []
        
        success = True
        for i in range(N):
            if time_remaining() < 10:
                success = False
                break
                
            res = place_circle_greedy(placed, W, H)
            if res is None:
                success = False
                break
            
            x, y, r = res
            placed.append((x, y, r))
            x_list.append(x)
            y_list.append(y)
            r_list.append(r)
        
        if not success:
            continue
            
        x = np.array(x_list)
        y = np.array(y_list)
        r = np.array(r_list)
        
        # Refine
        x_ref, y_ref, r_ref, score = refine_configuration(x, y, r, W, H, iterations=50)
        
        if score > best_overall_score:
            best_overall_score = score
            best_result = (x_ref, y_ref, r_ref, W, H)
            
    if best_result is None:
        # Fallback to a simple grid if greedy failed
        W, H = 1.0, 1.0
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        return np.column_stack([x, y, r])
        
    x, y, r, W, H = best_result
    return np.column_stack([x, y, r])
