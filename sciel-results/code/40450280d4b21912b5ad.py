"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using BFGS on g(x)=f(x)^2 with robust fallbacks.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
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
    
    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except Exception:
        return best_x
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return True
        return False
    
    update_best(best_x, best_fx)
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    # Phase 1: BFGS on g(x) = f(x)^2
    # BFGS maintains Hessian approximation H (scalar in 1D)
    # g(x) = f(x)^2, g'(x) = 2*f(x)*f'(x)
    # We use finite differences for g' and BFGS update for H
    
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        fx = best_fx
    
    if fx is None or math.isnan(fx) or math.isinf(fx):
        fx = best_fx
        if fx is None:
            return best_x
    
    # Initial Hessian approximation for g(x) = f(x)^2
    # g''(x) = 2*(f'(x))^2 + 2*f(x)*f''(x)
    # Start with H = 1.0 / max(g''_estimate, eps)
    h_fd = 1e-6 * max(1.0, abs(x))
    try:
        fx_plus = call_f(x + h_fd)
        fx_minus = call_f(x - h_fd)
    except Exception:
        fx_plus = fx
        fx_minus = fx
    
    if fx_plus is not None and fx_minus is not None and not math.isnan(fx_plus) and not math.isinf(fx_plus) and not math.isnan(fx_minus) and not math.isinf(fx_minus):
        fp = (fx_plus - fx_minus) / (2 * h_fd)
        fpp = (fx_plus - 2*fx + fx_minus) / (h_fd * h_fd)
        g_pp = 2 * fp * fp + 2 * fx * fpp
        H = 1.0 / max(abs(g_pp), 1e-12)
        if H > 1e6:
            H = 1e6
        if H < 1e-6:
            H = 1e-6
    else:
        H = 1.0
    
    # BFGS iterations on g(x) = f(x)^2
    prev_x = None
    prev_g = None
    prev_gp = None
    
    for iteration in range(60):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        
        # Compute g(x) and g'(x) using finite difference
        h_g = 1e-7 * max(1.0, abs(x))
        try:
            fx_curr = call_f(x)
        except Exception:
            break
        update_best(x, fx_curr)
        if fx_curr is None or math.isnan(fx_curr) or math.isinf(fx_curr):
            break
        
        g_val = fx_curr * fx_curr
        
        if abs(fx_curr) <= 1e-8:
            return x
        
        # Compute g'(x) = 2*f(x)*f'(x) using central difference
        try:
            fx_p = call_f(x + h_g)
            fx_m = call_f(x - h_g)
        except Exception:
            break
        
        if fx_p is None or fx_m is None or math.isnan(fx_p) or math.isinf(fx_p) or math.isnan(fx_m) or math.isinf(fx_m):
            break
        
        fp = (fx_p - fx_m) / (2 * h_g)
        gp = 2 * fx_curr * fp
        
        # BFGS update for H
        if prev_x is not None and prev_g is not None and prev_gp is not None:
            s = x - prev_x
            g_diff = gp - prev_gp
            if abs(s) > 1e-15 and abs(g_diff) > 1e-15:
                # BFGS update: H_new = H_old + (1 + H_old * g_diff / s) * s^2 / g_diff
                # In 1D: H_new = s^2 / (g_diff * (1 + H_old * g_diff / s))
                # Simplified: H_new = s^2 / g_diff * (1 + H_old * g_diff / s) / (1 + H_old * g_diff / s)
                # Actually in 1D BFGS: H_new = s^2 / g_diff
                # But we want to maintain positive definiteness
                H_candidate = s * s / g_diff
                if H_candidate > 0 and abs(H_candidate) < 1e12:
                    # Blend with previous H for stability
                    H = 0.5 * H + 0.5 * H_candidate
                    if H < 1e-8:
                        H = 1e-8
                    if H > 1e8:
                        H = 1e8
        
        prev_x = x
        prev_g = g_val
        prev_gp = gp
        
        # Newton step for g: x_new = x - g(x) / g'(x)
        # But g'(x) = 2*f(x)*f'(x), so step = -g/g' = -f^2/(2*f*f') = -f/(2*f')
        # This is equivalent to Newton on f with factor 1/2
        if abs(gp) > 1e-15:
            step = -g_val / gp
        else:
            # Fallback: use H * gp
            step = -H * gp
        
        # Limit step size
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        
        x_new = x + step
        
        # Line search: backtracking to ensure sufficient decrease in g
        alpha = 1.0
        c1 = 1e-4  # Armijo condition
        g_curr = g_val
        gp_curr = gp
        
        for ls_iter in range(20):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            
            x_try = x + alpha * step
            try:
                fx_try = call_f(x_try)
            except Exception:
                alpha *= 0.5
                continue
            
            update_best(x_try, fx_try)
            if fx_try is None or math.isnan(fx_try) or math.isinf(fx_try):
                alpha *= 0.5
                continue
            
            g_try = fx_try * fx_try
            
            # Armijo condition: g(x + alpha*step) <= g(x) + c1 * alpha * gp * step
            if g_try <= g_curr + c1 * alpha * gp_curr * step:
                break
            alpha *= 0.5
        
        if alpha < 1e-8:
            # Too many backtracks, try smaller step
            step *= 0.1
            x_new = x + step
            try:
                fx_new = call_f(x_new)
            except Exception:
                break
            update_best(x_new, fx_new)
            if fx_new is None or math.isnan(fx_new) or math.isinf(fx_new):
                break
        else:
            x_new = x + alpha * step
            try:
                fx_new = call_f(x_new)
            except Exception:
                break
            update_best(x_new, fx_new)
            if fx_new is None or math.isnan(fx_new) or math.isinf(fx_new):
                break
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        # Check for convergence
        if abs(step) < 1e-12 * max(1.0, abs(x_new)):
            break
        
        x = x_new
        fx = fx_new
    
    # Phase 2: If BFGS didn't converge, try bracketing + bisection
    if eval_count < MAX_EVALS and check_time() and abs(best_fx) > 1e-8:
        a, b, fa, fb = None, None, None, None
        step = 1e-4 * max(1.0, abs(best_x))
        for i in range(15):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            x_try = best_x + step
            try:
                fx_try = call_f(x_try)
            except Exception:
                step *= -2
                continue
            update_best(x_try, fx_try)
            if fx_try is None or math.isnan(fx_try) or math.isinf(fx_try):
                step *= -2
                continue
            if best_fx * fx_try < 0:
                a, b = best_x, x_try
                fa, fb = best_fx, fx_try
                break
            x_try2 = best_x - step
            try:
                fx_try2 = call_f(x_try2)
            except Exception:
                step *= 2
                continue
            update_best(x_try2, fx_try2)
            if fx_try2 is None or math.isnan(fx_try2) or math.isinf(fx_try2):
                step *= 2
                continue
            if best_fx * fx_try2 < 0:
                a, b = x_try2, best_x
                fa, fb = fx_try2, best_fx
                break
            step *= 2
        
        if a is not None and b is not None:
            for _ in range(80):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                m = (a + b) / 2.0
                try:
                    fm = call_f(m)
                except Exception:
                    break
                update_best(m, fm)
                if fm is None or math.isnan(fm) or math.isinf(fm):
                    break
                if abs(fm) <= 1e-8:
                    return m
                if fa * fm < 0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
    
    # Phase 3: Secant fallback
    if eval_count < MAX_EVALS and check_time() and abs(best_fx) > 1e-8:
        x0_sec = best_x
        try:
            fx0 = call_f(x0_sec)
        except Exception:
            fx0 = best_fx
        h = 1e-5 * max(1.0, abs(x0_sec))
        x1_sec = x0_sec + h
        try:
            fx1 = call_f(x1_sec)
        except Exception:
            fx1 = None
        if fx1 is not None and not math.isnan(fx1) and not math.isinf(fx1):
            for _ in range(40):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                if fx0 is None or fx1 is None:
                    break
                if abs(fx1) <= 1e-8:
                    return x1_sec
                if abs(fx0) <= 1e-8:
                    return x0_sec
                if fx0 == fx1:
                    break
                x2_sec = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
                try:
                    fx2 = call_f(x2_sec)
                except Exception:
                    break
                update_best(x2_sec, fx2)
                if fx2 is None or math.isnan(fx2) or math.isinf(fx2):
                    break
                if abs(fx2) <= 1e-8:
                    return x2_sec
                x0_sec, fx0 = x1_sec, fx1
                x1_sec, fx1 = x2_sec, fx2
    
    return best_x
# EVOLVE-BLOCK-END
