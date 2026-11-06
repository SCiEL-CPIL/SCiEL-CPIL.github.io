"""Root finding via hybrid derivative-free and secant polishing."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a combination of global exploration, Nelder-Mead minimization of f(x)^2,
    and secant polishing for ill-conditioned cases.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return float('inf')
        if time.time() > deadline:
            return float('inf')
        eval_count += 1
        try:
            val = f(x)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return float('inf')
            return val
        except (OverflowError, ZeroDivisionError, ValueError):
            return float('inf')
    
    def g(x):
        """Objective: f(x)^2"""
        fx = eval_f(x)
        if fx == float('inf'):
            return float('inf')
        return fx * fx
    
    # Track best point found
    best_x = float(x0)
    best_f = eval_f(best_x)
    best_g = best_f * best_f if best_f != float('inf') else float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_g
        if fx != float('inf') and abs(fx) < abs(best_f):
            best_x = x
            best_f = fx
            best_g = fx * fx
    
    # Check if already solved
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Phase 1: Nelder-Mead simplex to minimize f(x)^2
    # Initialize simplex around x0
    x0 = float(x0)
    scale = max(1.0, abs(x0)) * 1e-3
    simplex = [
        (x0, best_g),
        (x0 + scale, g(x0 + scale)),
    ]
    
    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    nm_iterations = 0
    max_nm_iter = 200
    
    while nm_iterations < max_nm_iter:
        if time.time() > deadline or eval_count >= max_evals:
            break
        nm_iterations += 1
        
        # Sort simplex by g value
        simplex.sort(key=lambda t: t[1])
        x_worst, g_worst = simplex[-1]
        x_best, g_best = simplex[0]
        x_second, g_second = simplex[1]
        
        # Update best
        if g_best < best_g:
            best_g = g_best
            best_x = x_best
            best_f = math.sqrt(g_best) if g_best >= 0 else 0.0
            if abs(best_f) <= 1e-8:
                return best_x
        
        # Centroid of all points except worst
        x_c = x_best  # In 1D, centroid of the other point
        
        # Reflection
        x_r = x_c + alpha * (x_c - x_worst)
        g_r = g(x_r)
        update_best(x_r, math.sqrt(g_r) if g_r != float('inf') and g_r >= 0 else float('inf'))
        
        if g_r < g_best:
            # Expansion
            x_e = x_c + gamma * (x_c - x_worst)
            g_e = g(x_e)
            update_best(x_e, math.sqrt(g_e) if g_e != float('inf') and g_e >= 0 else float('inf'))
            if g_e < g_r:
                simplex[-1] = (x_e, g_e)
            else:
                simplex[-1] = (x_r, g_r)
        elif g_r < g_second:
            simplex[-1] = (x_r, g_r)
        else:
            # Contraction
            x_cw = x_c + rho * (x_worst - x_c)
            g_cw = g(x_cw)
            update_best(x_cw, math.sqrt(g_cw) if g_cw != float('inf') and g_cw >= 0 else float('inf'))
            if g_cw < g_worst:
                simplex[-1] = (x_cw, g_cw)
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    x_s = x_best + sigma * (simplex[i][0] - x_best)
                    g_s = g(x_s)
                    update_best(x_s, math.sqrt(g_s) if g_s != float('inf') and g_s >= 0 else float('inf'))
                    simplex[i] = (x_s, g_s)
        
        # Check convergence
        if abs(g_worst - g_best) < 1e-14:
            break
    
    # Check if solved
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Phase 2: Try secant method from multiple starting points
    # Generate candidate pairs based on best point and its neighbors
    candidates = []
    base = best_x
    for offset in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, -1e-2, -1e-3, -1e-4, -1e-5, -1e-6]:
        candidates.append(base + offset * max(1.0, abs(base)))
    
    # Also try original x0 and neighbors
    candidates.append(x0)
    candidates.append(x0 + 1e-3 * max(1.0, abs(x0)))
    candidates.append(x0 - 1e-3 * max(1.0, abs(x0)))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = round(c, 15)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Try secant from best candidate pairs
    for i in range(min(len(unique_candidates), 10)):
        if time.time() > deadline or eval_count >= max_evals:
            break
        if abs(best_f) <= 1e-8:
            return best_x
        
        x1 = unique_candidates[i]
        # Find a second point
        x2 = None
        for j in range(len(unique_candidates)):
            if i != j and abs(unique_candidates[j] - x1) > 1e-12:
                x2 = unique_candidates[j]
                break
        if x2 is None:
            x2 = x1 + 1e-4 * max(1.0, abs(x1))
        
        # Secant method
        fx1 = eval_f(x1)
        update_best(x1, fx1)
        if abs(fx1) <= 1e-8:
            return x1
        if fx1 == float('inf'):
            continue
        
        fx2 = eval_f(x2)
        update_best(x2, fx2)
        if abs(fx2) <= 1e-8:
            return x2
        if fx2 == float('inf'):
            continue
        
        for _ in range(30):
            if time.time() > deadline or eval_count >= max_evals:
                break
            if abs(fx2 - fx1) < 1e-15:
                break
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            update_best(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            if fx_new == float('inf'):
                break
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
    
    # Phase 3: If still not solved, try a few more exploratory steps
    if abs(best_f) > 1e-8 and time.time() < deadline and eval_count < max_evals:
        # Try bisection-like approach around best_x if sign change detected
        for scale_factor in [1e-1, 1e-2, 1e-3, 1e-4]:
            if time.time() > deadline or eval_count >= max_evals:
                break
            x_left = best_x - scale_factor * max(1.0, abs(best_x))
            x_right = best_x + scale_factor * max(1.0, abs(best_x))
            fx_left = eval_f(x_left)
            update_best(x_left, fx_left)
            if abs(fx_left) <= 1e-8:
                return x_left
            fx_right = eval_f(x_right)
            update_best(x_right, fx_right)
            if abs(fx_right) <= 1e-8:
                return x_right
            if fx_left == float('inf') or fx_right == float('inf'):
                continue
            if fx_left * fx_right < 0:
                # Bisection
                a, fa = x_left, fx_left
                b, fb = x_right, fx_right
                for _ in range(50):
                    if time.time() > deadline or eval_count >= max_evals:
                        break
                    m = (a + b) / 2
                    fm = eval_f(m)
                    update_best(m, fm)
                    if abs(fm) <= 1e-8:
                        return m
                    if fm == float('inf'):
                        break
                    if fa * fm <= 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
    
    return best_x
# EVOLVE-BLOCK-END
