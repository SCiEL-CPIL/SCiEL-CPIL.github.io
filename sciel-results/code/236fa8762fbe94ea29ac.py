import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_value = 0.0
    best_result = None

    def _is_valid(cx, cy, r, W, H, tol=1e-9):
        if W <= 0.01 or H <= 0.01:
            return False
        if np.any(cx - r < -tol) or np.any(cy - r < -tol):
            return False
        if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
            return False
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < r_sum[iu]**2 - tol):
            return False
        return True

    def _force_relax(x, y, r, W, H, max_iter=2000, dt_init=0.01):
        """
        Vectorized force-directed relaxation to resolve overlaps and boundary violations.
        Returns updated x, y, r (r is scaled down if necessary to fit, but we prefer to move).
        Actually, we want to maximize sum(r). So we should try to keep r large and move x,y.
        This function adjusts x,y to satisfy constraints given r, W, H.
        If it fails, it might reduce r slightly, but we'll handle that in the polish.
        """
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        dt = dt_init
        min_dt = 1e-6
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Boundary forces
            # Push away from left
            viol_left = np.maximum(r - x, 0)
            fx += viol_left * 100
            # Push away from right
            viol_right = np.maximum(x + r - W, 0)
            fx -= viol_right * 100
            # Push away from bottom
            viol_bottom = np.maximum(r - y, 0)
            fy += viol_bottom * 100
            # Push away from top
            viol_top = np.maximum(y + r - H, 0)
            fy -= viol_top * 100
            
            # Pairwise forces
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx*dx + dy*dy
            d = np.sqrt(np.maximum(d2, 1e-12))
            r_sum = r[:, None] + r[None, :]
            
            # Overlap amount
            overlap = np.maximum(r_sum - d, 0)
            
            if np.max(overlap) < 1e-9:
                # Check if we are still violating boundaries
                if np.max(viol_left) < 1e-9 and np.max(viol_right) < 1e-9 and \
                   np.max(viol_bottom) < 1e-9 and np.max(viol_top) < 1e-9:
                    break
            
            # Forces from overlap
            # Direction from j to i
            ux = dx / d
            uy = dy / d
            
            # Force magnitude proportional to overlap
            f_mag = overlap * 10.0
            
            # Sum forces for each i
            # fx[i] += sum_j f_mag[i,j] * ux[i,j]
            fx += np.sum(f_mag * ux, axis=1)
            fy += np.sum(f_mag * uy, axis=1)
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clip to box to prevent escaping
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            # Decay dt
            dt *= 0.99
            
        return x, y, r

    def _slsqp_polish(x0, y0, r0, W, H):
        """
        Polish a valid (or near valid) configuration using SLSQP with analytic gradients.
        Variables: x[0:N], y[0:N], r[0:N]
        """
        def objective(z):
            # z = [x..., y..., r...]
            r = z[2*N:3*N]
            return -np.sum(r)
            
        def objective_grad(z):
            grad = np.zeros(3*N)
            grad[2*N:3*N] = -1.0
            return grad
            
        constraints = []
        
        # Boundary constraints: x_i >= r_i, x_i <= W - r_i, etc.
        # g(z) >= 0
        # x_i - r_i >= 0
        # W - x_i - r_i >= 0
        # y_i - r_i >= 0
        # H - y_i - r_i >= 0
        
        # We can vectorize the constraint functions for speed, but SLSQP expects individual constraint dicts.
        # For 21 circles, 84 boundary constraints + 210 overlap constraints = 294 constraints.
        # This might be slow. Let's see.
        
        # Alternative: Penalty method with high weight? No, SLSQP is better for final polish.
        # To speed up, we can define the constraints more efficiently.
        
        # Let's define the constraints.
        # For efficiency, we might skip SLSQP if the force relaxation already gives a good result and just do a few steps of gradient ascent on r with projection.
        # But SLSQP is robust. Let's try it.
        
        # To avoid lambda closure issues and speed up, we pre-define indices.
        
        # Overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # This is non-smooth at d=0, but we start from a relaxed state so d > 0.
        
        # Given the time limit, 294 constraints might be too many for SLSQP to converge quickly.
        # Let's use a simpler local search: 
        # 1. Try to increase all r by epsilon.
        # 2. If invalid, identify conflicts.
        # 3. Move centers to resolve conflicts.
        # This is essentially what force relaxation does, but we want to *expand* r.
        
        # Strategy: 
        # 1. Force relax to get a valid packing with current r.
        # 2. Try to increase r by a small amount.
        # 3. Force relax again.
        # 4. Repeat.
        
        # This "expand and relax" loop is often very effective for packing.
        
        pass # We will implement this loop instead of SLSQP if it's faster.

    def _expand_and_relax(x, y, r, W, H, max_steps=1000):
        """
        Iteratively increase radii and relax positions.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # Determine max possible r based on area? No, just try to increase.
        # We want to maximize sum(r). 
        # We can try to increase all r by the same factor or individually.
        # Increasing all by same factor is a good heuristic.
        
        current_sum = np.sum(r)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Try to increase radii
            # Scale factor
            # We don't know the optimal scale, so we try a small increment.
            # Let's try to increase r by 1% each step.
            r_new = r * 1.01
            
            # Relax positions for new radii
            x_new, y_new, _ = _force_relax(x, y, r_new, W, H, max_iter=500, dt_init=0.01)
            
            # Check if the relaxation succeeded (i.e., is valid)
            if _is_valid(x_new, y_new, r_new, W, H):
                x = x_new
                y = y_new
                r = r_new
                # If we made progress, continue
            else:
                # If it failed, try a smaller increment
                # Or break if we can't increase anymore
                # We can try to find the max factor that works via binary search, but that's expensive.
                # Let's just try a smaller factor.
                r_new = r * 1.005
                x_new, y_new, _ = _force_relax(x, y, r_new, W, H, max_iter=500, dt_init=0.01)
                if _is_valid(x_new, y_new, r_new, W, H):
                    x = x_new
                    y = y_new
                    r = r_new
                else:
                    # Can't increase anymore with this strategy
                    break
                    
        return x, y, r

    # Main Loop: Try different aspect ratios and initial guesses
    
    # Aspect ratios W (H = 2-W)
    # Optimal is likely near square or slightly elongated.
    W_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Initial guesses
    def _get_guesses(W, H):
        guesses = []
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        z_grid = np.zeros((N, 3))
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * W
            cy = (row + 0.5) / side * H
            r0 = min(W / side, H / side) * 0.45
            z_grid[i] = [cx, cy, r0]
        guesses.append(z_grid)
        
        # 2. Random
        rng = np.random.default_rng(42 + int(W*100))
        for _ in range(3):
            z_rand = np.zeros((N, 3))
            for i in range(N):
                r0 = rng.uniform(0.05, 0.15)
                cx = rng.uniform(r0, W - r0)
                cy = rng.uniform(r0, H - r0)
                z_rand[i] = [cx, cy, r0]
            guesses.append(z_rand)
            
        return guesses

    for W in W_values:
        H = 2.0 - W
        if H < 0.5: continue
        
        guesses = _get_guesses(W, H)
        
        for z0 in guesses:
            if time.time() > deadline - 10:
                break
                
            x0 = z0[:, 0]
            y0 = z0[:, 1]
            r0 = z0[:, 2]
            
            # Phase 1: Force Relax to get a valid packing
            x, y, r = _force_relax(x0, y0, r0, W, H, max_iter=1000)
            
            # Check validity
            if not _is_valid(x, y, r, W, H):
                # If still invalid, shrink radii until valid
                # This is a fallback
                scale = 1.0
                for _ in range(20):
                    scale *= 0.9
                    r_trial = r * scale
                    x_trial, y_trial, _ = _force_relax(x, y, r_trial, W, H, max_iter=200)
                    if _is_valid(x_trial, y_trial, r_trial, W, H):
                        x = x_trial
                        y = y_trial
                        r = r_trial
                        break
                else:
                    continue # Skip this guess if we can't make it valid
            
            # Phase 2: Expand and Relax
            x, y, r = _expand_and_relax(x, y, r, W, H, max_steps=200)
            
            # Final check
            if _is_valid(x, y, r, W, H):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = np.column_stack((x, y, r))
                    
    if best_result is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([cx, cy, r])
        return np.array(cells[:N])
        
    return best_result
