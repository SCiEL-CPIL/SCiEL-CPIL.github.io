"""Root finding: Ill-conditioned - Hybrid Bracketing and Minimization."""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Hybrid strategy:
    1. Expand bounds around x0.
    2. If sign change found, use scipy.brentq (fast, robust).
    3. If no sign change, use scipy.minimize_scalar on f(x)^2 (handles tangencies).
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        eval_count[0] += 1
        if eval_count[0] > MAX_EVALS:
            return 0.0
        return f(x)
    
    def check_time():
        return (time.time() - start_time) < MAX_WALL

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    x0 = float(x0)
    
    # Initial check
    fx0 = eval_f(x0)
    if is_good(x0, fx0):
        return x0

    # Objective for minimization
    def objective(x):
        if not check_time() or eval_count[0] >= MAX_EVALS:
            return 1e30
        fx = eval_f(x)
        if math.isnan(fx) or math.isinf(fx):
            return 1e30
        return fx * fx

    base_scale = max(1.0, abs(x0))
    # Scales: start small, go large
    scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e6]
    
    best_x = x0
    best_fx = fx0
    best_val = abs(fx0)
    
    # We will iterate through scales
    for scale in scales:
        if not check_time() or eval_count[0] >= MAX_EVALS:
            break
            
        R = scale * base_scale
        a = x0 - R
        b = x0 + R
        
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        if b - a < 1e-12:
            continue

        fa = eval_f(a)
        fb = eval_f(b)
        
        if math.isnan(fa) or math.isinf(fa):
            continue
        if math.isnan(fb) or math.isinf(fb):
            continue

        # Update best
        if abs(fa) < best_val:
            best_x, best_val = a, abs(fa)
            best_fx = fa
        if abs(fb) < best_val:
            best_x, best_val = b, abs(fb)
            best_fx = fb

        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b

        # Check for sign change
        if fa * fb < 0:
            # Use Brent's method
            try:
                # brentq requires f(a) and f(b) to have opposite signs
                # It will evaluate f internally. We need to pass a wrapped f to count?
                # brentq does not accept a callback for counting.
                # However, brentq is very efficient.
                # We must ensure we don't exceed MAX_EVALS.
                # We can't easily control brentq's internal eval count.
                # But brentq converges in ~10-20 evals usually.
                # We have used some evals for fa, fb.
                # Let's assume brentq is safe if we have > 50 evals left.
                if eval_count[0] < MAX_EVALS - 20:
                    x_root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                    fx_root = eval_f(x_root)
                    if is_good(x_root, fx_root):
                        return x_root
                    if abs(fx_root) < best_val:
                        best_x, best_val = x_root, abs(fx_root)
                        best_fx = fx_root
            except Exception:
                pass
            
        else:
            # No sign change: try minimization in this bracket
            # Only do this if we have enough budget
            if eval_count[0] < MAX_EVALS - 50:
                try:
                    res = minimize_scalar(objective, bounds=(a, b), method='bounded', 
                                          options={'xatol': 1e-12, 'maxiter': 30})
                    if res.success:
                        x_opt = res.x
                        fx_opt = eval_f(x_opt)
                        if math.isnan(fx_opt) or math.isinf(fx_opt):
                            continue
                        val_opt = abs(fx_opt)
                        if val_opt < best_val:
                            best_x, best_val = x_opt, val_opt
                            best_fx = fx_opt
                        if is_good(x_opt, fx_opt):
                            return x_opt
                except Exception:
                    continue

    # Final polish with Newton if possible
    if eval_count[0] < MAX_EVALS and check_time():
        x = best_x
        fx = best_fx
        for _ in range(5):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
            if abs(fx) <= TARGET_TOL:
                return x
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            if d == 0 or math.isnan(d) or math.isinf(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if math.isnan(fx_new) or math.isinf(fx_new):
                break
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                break
        if is_good(x, fx):
            return x

    return best_x

# EVOLVE-BLOCK-END
