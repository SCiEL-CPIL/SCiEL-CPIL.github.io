"""Root finding for ill-conditioned problems using Nelder-Mead minimization."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 by minimizing f(x)^2 using Nelder-Mead.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    # Objective function to minimize: |f(x)|^2
    def objective(x):
        try:
            fx = call_f(x)
            if not math.isfinite(fx):
                return float('inf')
            return fx * fx
        except:
            return float('inf')

    # Track best point found globally
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
        if not math.isfinite(best_fx):
            best_fx = float('inf')
    except:
        best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    if abs(best_fx) <= 1e-8:
        return best_x

    # Nelder-Mead Simplex Method for 1D
    # Initial simplex: x0 and x0 + initial_step
    scale = max(1.0, abs(x0))
    initial_step = 1e-3 * scale
    
    # Vertices: list of (x, f_value)
    # Start with two points to form a 1D simplex
    x1 = best_x
    f1 = objective(x1)
    
    x2 = best_x + initial_step
    f2 = objective(x2)
    
    # If initial points are bad, try a larger step or opposite direction
    if f2 >= f1 and f2 >= float('inf'):
         x2 = best_x - initial_step
         f2 = objective(x2)
         
    # Ensure x1 is the better point
    if f2 < f1:
        x1, x2 = x2, x1
        f1, f2 = f2, f1
        
    # Main loop
    # Standard Nelder-Mead parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink
    
    # We will run for a fixed number of iterations or until eval limit
    # In 1D, the simplex is just two points. 
    # Actually, standard Nelder-Mead in 1D is a bit degenerate. 
    # A 1D simplex has 2 vertices.
    # Operations:
    # 1. Reflect the worst point (x2) across the best (x1).
    #    x_r = x1 - alpha * (x2 - x1) = (1+alpha)*x1 - alpha*x2
    # 2. If f(x_r) < f(x1), try expansion: x_e = x1 + gamma*(x_r - x1)
    # 3. If f(x_r) > f(x2), try contraction: x_c = x1 + rho*(x2 - x1)
    # 4. If all fail, shrink: x2 = x1 + sigma*(x2 - x1)
    
    # To make it more robust, we can also allow "restarts" or larger jumps if stuck.
    # But let's stick to standard NM first.
    
    # Initial state:
    # Best: x1, Worst: x2
    
    for i in range(300): # Max iterations for NM
        if not check_time() or eval_count >= MAX_EVALS:
            break
            
        # Check convergence
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        # If the function values are very close, we might be at a minimum or flat
        if abs(f1 - f2) < 1e-15 * max(1.0, abs(f1)):
             # Try a larger perturbation to escape local minimum/flat region
             # This is a heuristic restart
             new_x = x1 + 10 * initial_step * (1 if x2 > x1 else -1)
             new_f = objective(new_x)
             if new_f < f2:
                 x2, f2 = new_x, new_f
             elif new_f < f1:
                 x1, x2 = new_x, x1
                 f1, f2 = new_f, f2
             else:
                 # Try other direction
                 new_x = x1 - 10 * initial_step * (1 if x2 > x1 else -1)
                 new_f = objective(new_x)
                 if new_f < f2:
                     x2, f2 = new_x, new_f
                 elif new_f < f1:
                     x1, x2 = new_x, x1
                     f1, f2 = new_f, f2
                 else:
                     break # Stuck
            
        # Identify best and worst
        if f1 <= f2:
            x_best, f_best = x1, f1
            x_worst, f_worst = x2, f2
        else:
            x_best, f_best = x2, f2
            x_worst, f_worst = x1, f1
            
        # Centroid (in 1D, it's just the best point since we only have 2 points? 
        # No, centroid of the other points. If we have 2 points, the "other" point is the best one.
        # So centroid c = x_best)
        c = x_best
        
        # 1. Reflection
        x_r = c - alpha * (x_worst - c)
        f_r = objective(x_r)
        update_best(x_r, math.sqrt(f_r) if f_r < float('inf') else float('inf'))
        
        if f_r < f_best:
            # Try Expansion
            x_e = c + gamma * (x_r - c)
            f_e = objective(x_e)
            update_best(x_e, math.sqrt(f_e) if f_e < float('inf') else float('inf'))
            
            if f_e < f_r:
                # Accept expansion
                x_worst, f_worst = x_e, f_e
            else:
                # Accept reflection
                x_worst, f_worst = x_r, f_r
        else:
            # Reflection not better than best
            if f_r < f_worst:
                # Accept reflection (it's better than worst, though not best)
                x_worst, f_worst = x_r, f_r
            else:
                # Contraction
                # Inside contraction (towards best)
                x_c = c + rho * (x_worst - c)
                f_c = objective(x_c)
                update_best(x_c, math.sqrt(f_c) if f_c < float('inf') else float('inf'))
                
                if f_c < f_worst:
                    x_worst, f_worst = x_c, f_c
                else:
                    # Shrink
                    # Shrink all points except best towards best
                    # In 1D, we just move the worst point closer to best
                    x_worst = c + sigma * (x_worst - c)
                    f_worst = objective(x_worst)
                    update_best(x_worst, math.sqrt(f_worst) if f_worst < float('inf') else float('inf'))
                    
                    # If interval is too small, break
                    if abs(x_worst - x_best) < 1e-12 * max(1.0, abs(x_best)):
                        break

        # Update x1, x2 based on which is better
        if f_worst < f_best:
            x1, x2 = x_worst, x_best
            f1, f2 = f_worst, f_best
        else:
            x1, x2 = x_best, x_worst
            f1, f2 = f_best, f_worst

    # If NM didn't find a root, try a few secant steps from the best points
    if eval_count < MAX_EVALS and check_time():
        # Use x1 and x2 as secant points
        # Ensure we have finite values
        if math.isfinite(f1) and math.isfinite(f2) and abs(f1 - f2) > 1e-15:
            x_a, f_a = x1, math.sqrt(f1) * (1 if f1 > 0 else -1) # Recover sign? 
            # Actually, objective is f^2, so we lost sign info for secant.
            # Secant requires f(x), not f(x)^2.
            # We can re-evaluate f at the best points to get signs if needed, 
            # but we already have f values stored in best_fx? No, best_fx is just the min abs.
            # Let's just do a local bisection if we can bracket, or a few Newton steps with FD.
            
            # Let's try Newton from best_x with a small step
            x_newton = best_x
            fx_newton = best_fx
            for k in range(20):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                if abs(fx_newton) <= 1e-8:
                    return x_newton
                h = 1e-5 * max(1.0, abs(x_newton))
                try:
                    fx_plus = call_f(x_newton + h)
                    d = (fx_plus - fx_newton) / h
                    if math.isfinite(d) and abs(d) > 1e-15:
                        step = fx_newton / d
                        # Damped Newton
                        for damp in [1.0, 0.5, 0.25]:
                            x_try = x_newton - damp * step
                            fx_try = call_f(x_try)
                            if math.isfinite(fx_try) and abs(fx_try) < abs(fx_newton):
                                x_newton = x_try
                                fx_newton = fx_try
                                update_best(x_newton, fx_try)
                                break
                except:
                    break

    return best_x
# EVOLVE-BLOCK-END
