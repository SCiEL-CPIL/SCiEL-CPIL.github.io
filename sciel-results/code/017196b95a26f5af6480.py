"""Circle packing in perimeter-4 rectangle: Simulated Annealing + SLSQP Refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_value = 0.0
    best_solution = None
    
    # Helper: Check validity strictly
    def check_validity(x, y, r, w, h, tol=1e-8):
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x - r < -tol) or np.any(x + r > w + tol):
            return False, 0.0
        if np.any(y - r < -tol) or np.any(y + r > h + tol):
            return False, 0.0
        # Vectorized pairwise check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx*dx + dy*dy
        r_sum = r[:, None] + r[None, :]
        # Only check upper triangle to save time, but np.triu_indices is fast
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < (r_sum[iu] - tol)**2):
            return False, 0.0
        return True, float(np.sum(r))

    # Objective for SLSQP: Minimize negative sum of radii + penalties
    def objective(z):
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w
        
        # Hard constraints penalty (if SLSQP bounds aren't tight enough or for smoothness)
        if w < 0.1 or h < 0.1:
            return 1e10
        if np.any(r < 0):
            return 1e10
            
        val = -np.sum(r)
        
        # Wall violations
        wall_pen = 0.0
        # x - r < 0 => r - x > 0
        over_x_left = np.maximum(r - x, 0)
        over_x_right = np.maximum(x + r - w, 0)
        over_y_bottom = np.maximum(r - y, 0)
        over_y_top = np.maximum(y + r - h, 0)
        wall_pen = np.sum(over_x_left**2 + over_x_right**2 + over_y_bottom**2 + over_y_top**2)
        
        # Overlap violations
        # d < r_i + r_j => r_i + r_j - d > 0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[:, None] + r[None, :]
        
        # Only upper triangle
        iu = np.triu_indices(N, k=1)
        overlap = np.maximum(r_sum[iu] - d[iu], 0)
        overlap_pen = np.sum(overlap**2)
        
        # Weight penalties heavily
        penalty_weight = 1000.0
        return val + penalty_weight * (wall_pen + overlap_pen)

    def slsqp_refine(x0, y0, r0, w0, time_limit):
        """Refine a valid or near-valid configuration using SLSQP."""
        z0 = np.concatenate([x0, y0, r0, [w0]])
        
        # Bounds
        # x in [0, w], y in [0, h], r in [0, max_possible]
        # Since w and h are variables, bounds are tricky.
        # We bound r to a reasonable max (e.g., 0.5) and x,y to [0, 2]
        # and rely on penalties for tight constraints.
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((1e-4, 0.5)) # r
        bounds.append((0.2, 1.8)) # w (h = 2-w, so h in [0.2, 1.8])
        
        # Constraints for SLSQP (optional, penalty method is often more robust for non-convex)
        # We'll stick to penalty in objective for robustness against local minima traps in constraints.
        
        try:
            res = minimize(
                objective, 
                z0, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < 1e6: # If it didn't blow up
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                w = res.x[3*N]
                h = 2.0 - w
                
                valid, val = check_validity(x, y, r, w, h)
                if valid:
                    return x, y, r, w, h, val
        except Exception:
            pass
        return None

    def simulated_annealing(x0, y0, r0, w0, time_budget):
        """
        SA on the penalty landscape.
        Moves: Perturb x, y, r, w.
        Acceptance: Metropolis criterion on the objective (penalized sum of radii).
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        w = w0.copy()
        
        # Initial temperature
        T = 0.05
        cooling_rate = 0.995
        min_T = 1e-6
        
        rng = np.random.default_rng(42)
        
        # Compute initial objective
        z0 = np.concatenate([x, y, r, [w]])
        current_obj = objective(z0)
        current_val = 0.0
        if check_validity(x, y, r, w, 2.0-w)[0]:
             current_val = np.sum(r)
        
        best_obj = current_obj
        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w.copy()
        best_val = current_val
        
        start_sa = time.time()
        no_improve_steps = 0
        max_no_improve = 1000
        
        while time.time() < time_budget and T > min_T:
            # Generate move
            move_type = rng.integers(0, 4)
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            new_w = w
            
            if move_type == 0: # Move x
                idx = rng.integers(0, N)
                new_x[idx] += rng.normal(0, 0.02)
            elif move_type == 1: # Move y
                idx = rng.integers(0, N)
                new_y[idx] += rng.normal(0, 0.02)
            elif move_type == 2: # Change radius
                idx = rng.integers(0, N)
                new_r[idx] = np.maximum(1e-4, new_r[idx] + rng.normal(0, 0.01))
            else: # Change aspect ratio
                new_w = np.clip(w + rng.normal(0, 0.02), 0.2, 1.8)
            
            # Ensure basic bounds for x,y,r to avoid insane objective values
            new_x = np.clip(new_x, 0, 2)
            new_y = np.clip(new_y, 0, 2)
            new_r = np.clip(new_r, 1e-4, 0.5)
            
            z_new = np.concatenate([new_x, new_y, new_r, [new_w]])
            new_obj = objective(z_new)
            
            # Metropolis acceptance
            delta_obj = new_obj - current_obj
            if delta_obj < 0:
                accept = True
            else:
                if T > 0:
                    accept = rng.random() < np.exp(-delta_obj / T)
                else:
                    accept = False
            
            if accept:
                x, y, r, w = new_x, new_y, new_r, new_w
                current_obj = new_obj
                
                # Check validity occasionally to update best
                if rng.random() < 0.1:
                    valid, val = check_validity(x, y, r, w, 2.0-w)
                    if valid and val > best_val:
                        best_val = val
                        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
                    
                if no_improve_steps > max_no_improve:
                    # Reset no_improve but keep going, maybe restart SA locally?
                    # For now, just continue cooling.
                    no_improve_steps = 0
                    T = max(T * 0.5, min_T) # Faster cool if stuck
            
            T *= cooling_rate
            
            # Periodic SLSQP refinement to escape SA's slow progress
            if no_improve_steps % 200 == 0 and time.time() < time_budget - 5:
                res = slsqp_refine(x, y, r, w, time.time() + 2)
                if res:
                    rx, ry, rr, rw, rh, rval = res
                    if rval > best_val:
                        best_val = rval
                        best_x, best_y, best_r, best_w = rx, ry, rr, rw
                        x, y, r, w = rx, ry, rr, rw
                        current_obj = objective(np.concatenate([x, y, r, [w]]))
                        no_improve_steps = 0
        
        return best_x, best_y, best_r, best_w, best_val

    # Initial Guesses
    guesses = []
    side = int(np.ceil(np.sqrt(N)))
    
    # 1. Grid-based starts with varying aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3: continue
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0/side, h0/side) * 0.45
        guesses.append((x, y, r, w0))
        
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        w0 = rng.uniform(0.5, 1.5)
        h0 = 2.0 - w0
        x = rng.uniform(0.1, w0-0.1, N)
        y = rng.uniform(0.1, h0-0.1, N)
        r = rng.uniform(0.05, 0.12, N)
        guesses.append((x, y, r, w0))
        
    # Run SA + SLSQP on each guess
    for i, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        # Give each guess a portion of the time
        time_per_guess = max(5.0, (deadline - 10 - time.time()) / (len(guesses) - i))
        
        # First, try to fix overlaps with a quick force relaxation to give SA a better start
        # Simple force relax
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        w = w0
        h = 2.0 - w
        
        for _ in range(100):
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall forces
            fx -= np.maximum(r - x, 0) * 100
            fx += np.maximum(x + r - w, 0) * 100
            fy -= np.maximum(r - y, 0) * 100
            fy += np.maximum(y + r - h, 0) * 100
            
            # Pair forces
            for a in range(N):
                for b in range(a+1, N):
                    dx = x[a] - x[b]
                    dy = y[a] - y[b]
                    d = np.sqrt(dx*dx + dy*dy)
                    rs = r[a] + r[b]
                    if d < rs and d > 1e-6:
                        f = (rs - d) * 10
                        fx[a] += (dx/d)*f
                        fy[a] += (dy/d)*f
                        fx[b] -= (dx/d)*f
                        fy[b] -= (dy/d)*f
            
            x += 0.01 * fx
            y += 0.01 * fy
            x = np.clip(x, r, w-r)
            y = np.clip(y, r, h-r)
            
        # Check if valid after relax
        valid, val = check_validity(x, y, r, w, h)
        
        # Run SA
        res = simulated_annealing(x, y, r, w, min(time.time() + time_per_guess, deadline - 5))
        if res:
            bx, by, br, bw, bval = res
            if bval > best_value:
                best_value = bval
                best_solution = (bx, by, br, bw)
                
        # Final SLSQP polish on the best found so far
        if best_solution and time.time() < deadline - 5:
            bx, by, br, bw = best_solution
            res = slsqp_refine(bx, by, br, bw, time.time() + 5)
            if res:
                rx, ry, rr, rw, rh, rval = res
                if rval > best_value:
                    best_value = rval
                    best_solution = (rx, ry, rr, rw)

    if best_solution is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.01
        return np.column_stack((x, y, r))
        
    x, y, r, w = best_solution
    h = 2.0 - w
    
    # Final safety shrink
    for _ in range(10):
        r_trial = r * 0.9999
        if check_validity(x, y, r_trial, w, h)[0]:
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
