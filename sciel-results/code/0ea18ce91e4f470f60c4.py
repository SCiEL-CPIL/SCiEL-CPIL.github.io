"""Circle packing in perimeter-4 rectangle: Global Differential Evolution."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _objective(z):
    """
    Objective function for differential evolution.
    z[0] = w (width)
    z[1] = h (height)
    z[2:2+N] = x centers
    z[2+N:2+2*N] = y centers
    z[2+2*N:2+3*N] = radii
    
    Returns negative sum of radii (to minimize).
    Penalty for invalid configurations.
    """
    w = z[0]
    h = z[1]
    x = z[2:2+N]
    y = z[2+N:2+2*N]
    r = z[2+2*N:2+3*N]
    
    # Perimeter constraint: 2(w+h) <= 4 => w+h <= 2
    # If violated, return high penalty
    if w + h > 2.0 + 1e-5:
        return 1e6
    
    # Boundary checks
    if np.any(x - r < 0) or np.any(x + r > w) or np.any(y - r < 0) or np.any(y + r > h):
        # Penalty proportional to violation magnitude
        pen = 0
        pen += np.sum(np.maximum(0, r - x))
        pen += np.sum(np.maximum(0, r - y))
        pen += np.sum(np.maximum(0, x + r - w))
        pen += np.sum(np.maximum(0, y + r - h))
        return 100 + pen
    
    # Overlap checks
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = r[i] + r[j]
            if d < min_d:
                penalty += (min_d - d) ** 2
    
    if penalty > 1e-5:
        return 100 + penalty
        
    return -np.sum(r)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Define bounds for differential evolution
    # w + h <= 2. 
    # We search w in [0.5, 1.5] and h in [0.5, 1.5]
    # The constraint w+h<=2 will be handled by the objective penalty, 
    # but we can tighten bounds. 
    # Actually, let's just bound w and h independently and rely on penalty.
    # Better: w in [0.4, 1.6], h in [0.4, 1.6]
    
    # Variables: w, h, x_0...x_20, y_0...y_20, r_0...r_20
    # Total 2 + 63 = 65 variables
    
    bounds = [(0.4, 1.6)]  # w
    bounds.append((0.4, 1.6))  # h
    
    # Initial guesses for x, y, r are not needed for DE bounds, 
    # but we need bounds for x, y, r.
    # x in [0, w_max], y in [0, h_max], r in [0, min(w,h)/2]
    # Since w, h vary, we use loose bounds and let penalty handle it.
    # Max possible x is 1.6, max y is 1.6.
    # Max radius is 0.8 (if w=1.6, h=0.4, min is 0.4, r<=0.2. If w=1, h=1, r<=0.5)
    # Let's use r in [0.01, 0.5]
    
    for _ in range(N):
        bounds.append((0.0, 1.6)) # x
    for _ in range(N):
        bounds.append((0.0, 1.6)) # y
    for _ in range(N):
        bounds.append((0.01, 0.5)) # r
        
    # Differential Evolution settings
    # 'popsize' : 15 * len(bounds) is default. 65*15 = 975 individuals.
    # This might be too slow. Let's try popsize=10 or 12.
    # 'maxiter' : Number of iterations.
    
    # We need to be careful with time.
    # Let's run DE with a moderate popsize and maxiter, then polish.
    
    def _refine_with_slsqp(w, h, x, y, r):
        """
        Local refinement using SLSQP on fixed w, h.
        This helps escape shallow local minima of DE.
        """
        from scipy.optimize import minimize
        
        z0 = np.concatenate([x, y, r])
        
        def obj(z):
            ri = z[2*N:]
            return -np.sum(ri)
            
        def jac(z):
            g = np.zeros(3*N)
            g[2*N:] = -1.0
            return g
            
        constraints = []
        
        # Boundaries
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*N+i]}) # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda z, i=i: w - z[i] - z[2*N+i]}) # w - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i] - z[2*N+i]}) # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda z, i=i: h - z[N+i] - z[2*N+i]}) # h - y - r >= 0
            
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                def make_constraint(i, j):
                    def fun(z):
                        xi, yi, ri = z[i], z[N+i], z[2*N+i]
                        xj, yj, rj = z[j], z[N+j], z[2*N+j]
                        dx = xi - xj
                        dy = yi - yj
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < 1e-10:
                            return -100
                        return d - (ri + rj)
                    return fun
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
                
        bounds_local = []
        for i in range(N):
            bounds_local.append((0.0, w)) # x
        for i in range(N):
            bounds_local.append((0.0, h)) # y
        for i in range(N):
            bounds_local.append((0.01, min(w, h)/2)) # r
            
        try:
            res = minimize(obj, z0, method='SLSQP', jac=jac, bounds=bounds_local, 
                           constraints=constraints, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success:
                return res.x[:N], res.x[N:2*N], res.x[2*N:]
        except Exception:
            pass
        return x, y, r

    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
                
    # Run Differential Evolution
    # To ensure we finish in time, we monitor the callback.
    # DE callback is called after each iteration.
    
    de_state = {'best_val': 0.0, 'best_x': None}
    
    def de_callback(xk, convergence):
        nonlocal de_state
        w = xk[0]
        h = xk[1]
        x = xk[2:2+N]
        y = xk[2+N:2+2*N]
        r = xk[2+2*N:2+3*N]
        
        # Check validity and update best
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > de_state['best_val']:
                de_state['best_val'] = val
                de_state['best_x'] = xk.copy()
                update_best(x, y, r, w, h)
                
        # Check time
        if time.time() > deadline - 10:
            # Force stop by raising exception? No, DE doesn't support that cleanly.
            # We'll just check in the main loop after DE returns.
            # But DE might run long. We can't easily interrupt it from callback.
            # Instead, we set maxiter low enough or popsize low enough.
            pass

    # Strategy: Run DE for a few iterations with moderate popsize.
    # Then take the best result found and refine with SLSQP.
    
    # Let's try popsize=12, maxiter=20.
    # 65 variables. 12*65 = 780 individuals.
    # Each eval involves N^2/2 = 210 distance checks.
    # 780 * 210 * 20 = ~3.2M distance checks. Should be fast in NumPy?
    # My _objective is Python loop. That will be SLOW.
    
    # Optimization: Vectorize _objective.
    
    def _objective_fast(z):
        w = z[0]
        h = z[1]
        x = z[2:2+N]
        y = z[2+N:2+2*N]
        r = z[2+2*N:2+3*N]
        
        if w + h > 2.0 + 1e-5:
            return 1e6
        
        # Boundary violations
        # x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
        viol_x_min = np.maximum(0, r - x)
        viol_x_max = np.maximum(0, x + r - w)
        viol_y_min = np.maximum(0, r - y)
        viol_y_max = np.maximum(0, y + r - h)
        
        boundary_pen = np.sum(viol_x_min) + np.sum(viol_x_max) + np.sum(viol_y_min) + np.sum(viol_y_max)
        
        if boundary_pen > 1e-5:
            return 100 + boundary_pen
            
        # Overlap violations
        # Compute pairwise distances
        # Using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        min_d = r[:, None] + r[None, :]
        
        # Only check upper triangle
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        min_d_pairs = min_d[iu]
        
        overlaps = np.maximum(0, min_d_pairs - d_pairs)
        overlap_pen = np.sum(overlaps**2)
        
        if overlap_pen > 1e-5:
            return 100 + overlap_pen
            
        return -np.sum(r)

    # Re-define callback to use fast objective? No, callback gets xk, we can just evaluate.
    # But we already have _is_valid which is slow.
    # Let's rely on DE's internal tracking and just check at the end, or use a fast validity check in callback.
    
    def _is_valid_fast(x, y, r, w, h, tol=1e-6):
        if w + h > 2.0 + tol:
            return False
        if np.any(x - r < -tol) or np.any(x + r > w + tol) or np.any(y - r < -tol) or np.any(y + r > h + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        min_d = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d[iu] < min_d[iu] - tol):
            return False
        return True

    def de_callback_fast(xk, convergence):
        nonlocal best_value, best_result
        w = xk[0]
        h = xk[1]
        x = xk[2:2+N]
        y = xk[2+N:2+2*N]
        r = xk[2+2*N:2+3*N]
        
        if _is_valid_fast(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
                
        if time.time() > deadline - 20:
            # We can't stop DE, but we can return early from construct?
            # No, we are inside DE.
            # We'll just let it run and check time after.
            pass

    # Run DE
    # To be safe on time, use smaller popsize and maxiter.
    # popsize=10, maxiter=15
    try:
        result = differential_evolution(
            _objective_fast, 
            bounds, 
            maxiter=15, 
            popsize=10, 
            tol=1e-8, 
            seed=42, 
            callback=de_callback_fast,
            polish=False  # We will do our own polish
        )
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find a valid solution, or we have time, try to refine the best found
    if best_result is not None:
        x, y, r, w, h = best_result
        # Refine with SLSQP
        x_ref, y_ref, r_ref = _refine_with_slsqp(w, h, x, y, r)
        update_best(x_ref, y_ref, r_ref, w, h)
        
    # If still no valid result, or time remains, try a few random SLSQP starts
    if time.time() < deadline - 30:
        # Fallback: Random starts with SLSQP
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            w = np.random.uniform(0.8, 1.2)
            h = 2.0 - w
            x = np.random.uniform(0.1*w, 0.9*w, N)
            y = np.random.uniform(0.1*h, 0.9*h, N)
            r = np.full(N, 0.1)
            
            x_ref, y_ref, r_ref = _refine_with_slsqp(w, h, x, y, r)
            update_best(x_ref, y_ref, r_ref, w, h)
            
    if best_result is None:
        # Absolute fallback
        w, h = 1.0, 1.0
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = np.full(N, 0.05)
        best_result = (x, y, r, w, h)
        
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
