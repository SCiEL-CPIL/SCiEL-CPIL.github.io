import numpy as np
from scipy.optimize import minimize
import time
import itertools

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Safety margin for 600s limit
    
    def get_compositions(n):
        """Generate all compositions of n (ordered integer partitions)."""
        if n == 0:
            yield []
            return
        for i in range(1, n + 1):
            for rest in get_compositions(n - i):
                yield [i] + rest

    def solve_shelf_layout(comp):
        """
        Solve for optimal side lengths for a shelf layout defined by 'comp' (row sizes).
        Maximize sum(n_r * s_r)
        Subject to:
          s_r <= 1.0 / n_r  (width constraint for row r)
          sum(s_r) <= 1.0   (height constraint for total stack)
          s_r >= 0
        """
        k = len(comp)
        # Greedy allocation: prioritize rows with larger n_r (higher value per unit height)
        # Actually, to maximize sum(n_r * s_r), we should fill the height with the rows
        # that have the highest "efficiency" or simply solve the LP.
        # Since the constraint is sum(s_r) <= 1 and s_r <= 1/n_r,
        # and objective is sum(n_r * s_r),
        # The marginal value of increasing s_r by ds is n_r.
        # So we should increase s_r for the row with the largest n_r first, up to its limit 1/n_r.
        
        indices = sorted(range(k), key=lambda r: comp[r], reverse=True)
        s_vals = [0.0] * k
        remaining_h = 1.0
        
        for r in indices:
            max_s = 1.0 / comp[r]
            take = min(max_s, remaining_h)
            if take < 1e-9:
                break
            s_vals[r] = take
            remaining_h -= take
            
        val = sum(comp[r] * s_vals[r] for r in range(k))
        
        sides_list = []
        for r in range(k):
            for _ in range(comp[r]):
                sides_list.append(s_vals[r])
                
        return val, sides_list

    def generate_shelf_layout(comp, sides_list):
        """
        Generate centers and sides for a specific shelf layout.
        """
        centers = []
        curr_y = 0.0
        idx = 0
        
        for r, n_r in enumerate(comp):
            row_sides = sides_list[idx:idx + n_r]
            s_row = row_sides[0] # All squares in a row have same side
            
            curr_x = 0.0
            for i in range(n_r):
                cx = curr_x + s_row / 2
                cy = curr_y + s_row / 2
                centers.append([cx, cy])
                curr_x += s_row
                
            curr_y += s_row
            idx += n_r
            
        return np.array(centers, dtype=np.float64), np.array(sides_list, dtype=np.float64)

    def is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if n != 10:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    best_val = 0.0
    best_centers = None
    best_sides = None
    
    # Phase 1: Enumerate shelf layouts
    for comp in get_compositions(n):
        if time.time() - start_time > time_limit - 300: # Reserve time for optimization
            break
            
        val, sides_list = solve_shelf_layout(comp)
        if val <= best_val:
            continue
            
        centers, sides = generate_shelf_layout(comp, sides_list)
        
        if is_valid(centers, sides):
            best_val = val
            best_centers = centers
            best_sides = sides

    if best_centers is None:
        # Fallback
        best_centers = np.array([[0.5, 0.5]] * 10)
        best_sides = np.full(10, 0.01)

    # Phase 2: Local Refinement using Scipy
    # We optimize centers (20 vars) and sides (10 vars)
    # Total 30 vars.
    
    def objective(params):
        c = params[:2*n].reshape(n, 2)
        s = params[2*n:]
        
        # Penalty for going out of bounds
        pen = 0.0
        for i in range(n):
            if c[i, 0] - s[i]/2 < 0: pen += (c[i, 0] - s[i]/2)**2
            if c[i, 0] + s[i]/2 > 1: pen += (c[i, 0] + s[i]/2 - 1)**2
            if c[i, 1] - s[i]/2 < 0: pen += (c[i, 1] - s[i]/2)**2
            if c[i, 1] + s[i]/2 > 1: pen += (c[i, 1] + s[i]/2 - 1)**2
            
        # Penalty for overlaps
        # Overlap if |dx| < (si+sj)/2 AND |dy| < (si+sj)/2
        # We use a soft penalty: P = max(0, (si+sj)/2 - |dx|) * max(0, (si+sj)/2 - |dy|)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i, 0] - c[j, 0])
                dy = abs(c[i, 1] - c[j, 1])
                tx = (s[i] + s[j]) / 2.0
                ty = (s[i] + s[j]) / 2.0
                
                # If they overlap in x
                ox = tx - dx
                if ox > 0:
                    # If they overlap in y
                    oy = ty - dy
                    if oy > 0:
                        pen += ox * oy
                        
        return -np.sum(s) + 100.0 * pen

    # Initial parameters from best shelf layout
    x0 = np.concatenate([best_centers.flatten(), best_sides])
    
    # Bounds: Centers in [0,1], Sides in [0.001, 1.0]
    bounds = [(0, 1)] * (2*n) + [(0.001, 1.0)] * n

    # Try a few random perturbations of the best shelf layout to escape local minima
    np.random.seed(42)
    num_refine_starts = 5
    
    for k in range(num_refine_starts):
        if time.time() - start_time > time_limit - 60:
            break
            
        if k == 0:
            x_start = x0
        else:
            # Perturb centers slightly
            perturb = np.random.normal(0, 0.05, size=(n, 2))
            c_pert = best_centers + perturb
            # Clamp centers to [0.1, 0.9] to keep them safely inside
            c_pert = np.clip(c_pert, 0.05, 0.95)
            x_start = np.concatenate([c_pert.flatten(), best_sides])
            
        try:
            res = minimize(
                objective,
                x_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective(x0):
                c_opt = res.x[:2*n].reshape(n, 2)
                s_opt = res.x[2*n:]
                
                # Check validity of the optimized result
                if is_valid(c_opt, s_opt):
                    val_opt = np.sum(s_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_centers = c_opt
                        best_sides = s_opt
        except Exception:
            pass

    return best_centers, best_sides
