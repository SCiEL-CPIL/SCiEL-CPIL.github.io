"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    
    def objective(params, W, H, penalty_weight=100.0):
        """
        params: array of length 3*N: [x1, y1, r1, x2, y2, r2, ...]
        Returns: -sum(r_i) + penalty
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Objective: Minimize -sum(r)
        val = -np.sum(rs)
        
        # Penalty for overlaps
        # d_ij >= r_i + r_j
        # violation = max(0, r_i + r_j - d_ij)
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    val += penalty_weight * overlap * overlap
        
        # Penalty for bounds
        # 0 <= x <= W, 0 <= y <= H
        # Also r_i <= min(x_i, W-x_i, y_i, H-y_i)
        for i in range(N):
            # Bound violation in x
            v_left = rs[i] - xs[i]
            if v_left > 0:
                val += penalty_weight * v_left * v_left
            v_right = rs[i] - (W - xs[i])
            if v_right > 0:
                val += penalty_weight * v_right * v_right
                
            # Bound violation in y
            v_bottom = rs[i] - ys[i]
            if v_bottom > 0:
                val += penalty_weight * v_bottom * v_bottom
            v_top = rs[i] - (H - ys[i])
            if v_top > 0:
                val += penalty_weight * v_top * v_top
                
        return val

    def get_initial_guess(W, H):
        """Generate initial guess using a grid."""
        # 3x7 grid
        coords = []
        for r in range(3):
            for c in range(7):
                # Center in cell
                x = (c + 0.5) * W / 7.0
                y = (r + 0.5) * H / 3.0
                # Initial radius: small fraction of min cell size
                r_init = 0.4 * min(W/7.0, H/3.0)
                coords.append([x, y, r_init])
        
        return np.array(coords).flatten()

    best_value = 0.0
    best_solution = None
    
    # Try different aspect ratios
    # Perimeter 4 => 2(W+H) = 4 => W+H = 2
    aspect_ratios = [1.0, 1.5, 0.5, 2.0, 0.25]
    
    for ar in aspect_ratios:
        W = ar * (2.0 / (1.0 + ar))
        H = 2.0 - W
        
        x0 = get_initial_guess(W, H)
        
        # Bounds for Nelder-Mead are not directly supported in the same way, 
        # but we can rely on the penalty. 
        # However, to help convergence, we can use SLSQP if it's fast enough, 
        # but Nelder-Mead is robust for this non-smooth penalty.
        
        # To improve performance, we can run multiple restarts or just one good one.
        # Let's try 2 restarts per aspect ratio with slight noise.
        
        for restart in range(2):
            if restart > 0:
                # Add small noise
                noise = np.random.normal(0, 0.05, x0.shape)
                # Ensure positive radii in noise
                noise[2::3] = np.abs(noise[2::3])
                x0_noisy = x0 + noise
            else:
                x0_noisy = x0.copy()

            res = minimize(objective, x0_noisy, args=(W, H, 100.0), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success or res.fun < 0: # res.fun is -sum(r) + penalty
                # Check validity
                xs = res.x[0::3]
                ys = res.x[1::3]
                rs = res.x[2::3]
                
                # Validate: all r > 0, no overlaps, in bounds
                valid = True
                if np.any(rs <= 0):
                    valid = False
                if np.any(xs < 0) or np.any(xs > W) or np.any(ys < 0) or np.any(ys > H):
                    valid = False
                # Check overlaps
                for i in range(N):
                    if xs[i] < rs[i] or xs[i] > W - rs[i] or ys[i] < rs[i] or ys[i] > H - rs[i]:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if d < rs[i] + rs[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = np.sum(rs)
                    if val > best_value:
                        best_value = val
                        sol = np.zeros((N, 3))
                        sol[:, 0] = xs
                        sol[:, 1] = ys
                        sol[:, 2] = rs
                        best_solution = sol

    if best_solution is None:
        # Fallback to grid
        W = 1.0
        H = 1.0
        coords = []
        for r in range(3):
            for c in range(7):
                coords.append([(c + 0.5) / 7.0, (r + 0.5) / 3.0])
        coords = np.array(coords)
        xs = coords[:, 0] * W
        ys = coords[:, 1] * H
        r_bound_x = np.min(np.minimum(xs, W - xs))
        r_bound_y = np.min(np.minimum(ys, H - ys))
        r_bound = min(r_bound_x, r_bound_y)
        r_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_pair = min(r_pair, dist / 2.0)
        r = min(r_bound, r_pair)
        sol = np.zeros((N, 3))
        sol[:, 0] = xs
        sol[:, 1] = ys
        sol[:, 2] = r
        best_solution = sol

    return best_solution
# EVOLVE-BLOCK-END
