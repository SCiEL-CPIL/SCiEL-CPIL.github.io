import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 580  # Leave 20s margin
    
    # We use a variable transformation to ensure squares are inside [0,1]^2.
    # Variables: u_i, v_i in [0,1], s_i in [0.01, 1.0]
    # cx_i = s_i/2 + (1 - s_i) * u_i
    # cy_i = s_i/2 + (1 - s_i) * v_i
    # This guarantees cx_i in [s_i/2, 1 - s_i/2], so the square is inside.
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # u
        bounds.append((0.0, 1.0)) # v
        bounds.append((1e-4, 1.0)) # s

    def get_centers_sides(x):
        u = x[0:3*n:3]
        v = x[1:3*n:3]
        s = x[2:3*n:3]
        cx = s/2.0 + (1.0 - s) * u
        cy = s/2.0 + (1.0 - s) * v
        return cx, cy, s

    def penalty(x):
        cx, cy, s = get_centers_sides(x)
        pen = 0.0
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                S = (s[i] + s[j]) / 2.0
                if dx < S and dy < S:
                    # Penalty is proportional to the overlap area squared
                    pen += ((S - dx) * (S - dy))**2
        return pen

    def objective(x):
        cx, cy, s = get_centers_sides(x)
        pen = penalty(x)
        return -np.sum(s) + 1000.0 * pen

    # Initial guess: Grid layout
    # 10 squares. Let's try a 5x2 grid of small squares as a safe start.
    # Or better, let DE find it.
    # We'll run DE.
    
    best_x = None
    best_val = -np.inf
    best_obj = np.inf

    def callback(xk, convergence):
        nonlocal best_x, best_val, best_obj
        cx, cy, s = get_centers_sides(xk)
        pen = penalty(xk)
        if pen < 1e-8:
            val = np.sum(s)
            if val > best_val:
                best_val = val
                best_x = xk.copy()
        # Update best_obj for tracking, even if invalid? No, we want valid best.
        # But for stopping criteria, maybe track min objective.
        obj_val = objective(xk)
        if obj_val < best_obj:
            best_obj = obj_val
            # Only update best_x if it's valid? 
            # The problem requires valid. So we stick to valid updates for best_x.
        
        # Stop if time limit reached
        if time.time() - start_time > time_limit:
            return True # Stop optimization

    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8, 
            seed=42, 
            callback=callback,
            polish=False # We will do our own polish
        )
        
        # If DE found a valid solution, use it. Otherwise, check res.x
        if best_x is None:
            # Check if res.x is valid
            cx, cy, s = get_centers_sides(res.x)
            pen = penalty(res.x)
            if pen < 1e-8:
                best_x = res.x
                best_val = np.sum(s)
    except Exception as e:
        pass

    # Fallback: Simple valid configuration
    # 10 squares of side 0.1 in a 5x2 grid
    if best_x is None:
        u = np.zeros(n)
        v = np.zeros(n)
        s = np.full(n, 0.1)
        # 5 cols, 2 rows
        for i in range(n):
            c = i % 5
            r = i // 5
            # s=0.1, width 0.1.
            # cx = 0.05 + 0.9 * u. 
            # We want cx = 0.05, 0.15, 0.25, 0.35, 0.45
            # 0.05 = 0.05 + 0.9 * u => u=0
            # 0.15 = 0.05 + 0.9 * u => u = 0.1/0.9 = 1/9
            # cx = 0.05 + 0.9 * (c/4.0) ? No.
            # Let's just set u,v to place them.
            # cx = 0.05 + c * 0.1
            # cy = 0.05 + r * 0.1
            # 0.05 + c*0.1 = 0.05 + 0.9*u => u = c*0.1/0.9 = c/9
            # 0.05 + r*0.1 = 0.05 + 0.9*v => v = r/9
            u[i] = c / 9.0
            v[i] = r / 9.0
        
        best_x = np.array([u, v, s]).T.flatten()
        best_val = np.sum(s)

    # Local Polish
    # Use L-BFGS-B on the transformed variables
    # Objective: -sum(s) + penalty
    # We want to minimize this.
    
    def polish_objective(x):
        cx, cy, s = get_centers_sides(x)
        pen = penalty(x)
        return -np.sum(s) + 1000.0 * pen

    def polish_grad(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = polish_objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = polish_objective(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    try:
        res_polish = minimize(
            polish_objective, 
            best_x, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        cx, cy, s = get_centers_sides(res_polish.x)
        pen = penalty(res_polish.x)
        if pen < 1e-8:
            val = np.sum(s)
            if val > best_val:
                best_x = res_polish.x
                best_val = val
    except Exception as e:
        pass

    # Final Extraction
    cx, cy, s = get_centers_sides(best_x)
    
    # Safety check: If still invalid (overlap > tol), shrink squares
    # This is a last resort repair.
    # Check overlaps
    tol = 1e-7
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            S = (s[i] + s[j]) / 2.0
            if dx < S - tol and dy < S - tol:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Shrink all squares by 10% to ensure separation? 
        # Or just return the best valid one found? 
        # If best_x is invalid, we must return a valid one.
        # Let's use the fallback grid which is guaranteed valid.
        u = np.zeros(n)
        v = np.zeros(n)
        s = np.full(n, 0.1)
        for i in range(n):
            c = i % 5
            r = i // 5
            u[i] = c / 9.0
            v[i] = r / 9.0
        best_x = np.array([u, v, s]).T.flatten()
        cx, cy, s = get_centers_sides(best_x)

    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides
