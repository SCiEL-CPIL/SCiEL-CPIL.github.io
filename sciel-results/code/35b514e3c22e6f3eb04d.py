import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    x = float(x0)
    
    # Track evaluations
    eval_count = 0
    
    # Helper to evaluate f and track count
    def safe_eval(x_val):
        nonlocal eval_count
        if eval_count >= 499: # Leave 1 for safety
            return None
        fx_val = f(x_val)
        eval_count += 1
        if math.isnan(fx_val) or math.isinf(fx_val):
            return None
        return fx_val

    # Initial evaluation
    fx = safe_eval(x)
    if fx is None:
        return x
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return best_x

    # If initial point is bad, try to find a bracket first using expanding steps
    # This handles cases where x0 is far from root or in a region where secant might diverge
    bracket_found = False
    a, b = None, None
    fa, fb = None, None
    
    # Expand around x0 to find a sign change
    delta = 1e-3
    for _ in range(50):
        if time.time() > deadline:
            break
        if eval_count >= 490:
            break
            
        x_a = x - delta
        x_b = x + delta
        
        fa = safe_eval(x_a)
        fb = safe_eval(x_b)
        
        if fa is None or fb is None:
            delta *= 2.0
            continue
            
        if abs(fa) < best_abs_fx:
            best_x, best_fx, best_abs_fx = x_a, fa, abs(fa)
            if best_abs_fx <= 1e-8:
                return best_x
        if abs(fb) < best_abs_fx:
            best_x, best_fx, best_abs_fx = x_b, fb, abs(fb)
            if best_abs_fx <= 1e-8:
                return best_x
                
        if fa * fb < 0:
            a, b = x_a, x_b
            bracket_found = True
            break
            
        delta *= 2.0
        if delta > 1e10:
            break
            
    if bracket_found:
        try:
            root = brentq(f, a, b, maxiter=100, rtol=1e-12, xtol=1e-14)
            fr = f(root)
            if abs(fr) <= 1e-8:
                return root
            # Even if not perfect, it's likely better
            if abs(fr) < best_abs_fx:
                return root
        except Exception:
            pass
        return best_x

    # No bracket found. Use Secant Method.
    # We need two points. We have x0. Let's find a second point.
    # Try a small step in both directions to pick the better one for secant start.
    h = 1e-4 * max(1.0, abs(x))
    
    x1 = x + h
    fx1 = safe_eval(x1)
    
    if fx1 is None:
        x1 = x - h
        fx1 = safe_eval(x1)
        
    if fx1 is None:
        # If we can't even evaluate a neighbor, return best
        return best_x
        
    # Check if fx1 is better
    if abs(fx1) < best_abs_fx:
        best_x, best_fx, best_abs_fx = x1, fx1, abs(fx1)
        if best_abs_fx <= 1e-8:
            return best_x

    # Now we have (x, fx) and (x1, fx1).
    # If they have opposite signs, we have a bracket!
    if fx * fx1 < 0:
        try:
            root = brentq(f, x, x1, maxiter=100, rtol=1e-12, xtol=1e-14)
            fr = f(root)
            if abs(fr) <= 1e-8:
                return root
            if abs(fr) < best_abs_fx:
                return root
        except Exception:
            pass

    # Secant Loop
    # x_prev, fx_prev and x_curr, fx_curr
    x_prev, fx_prev = x, fx
    x_curr, fx_curr = x1, fx1
    
    # If fx_curr is not better than fx_prev, swap to keep fx_curr as the "newer" point 
    # but secant doesn't strictly require ordering, just distinct points.
    # However, for stability, if |fx_curr| > |fx_prev|, we might want to restart or be careful.
    
    for i in range(450):
        if time.time() > deadline:
            break
        if eval_count >= 498:
            break
            
        if abs(fx_curr) <= 1e-8:
            return x_curr
        if abs(fx_prev) <= 1e-8:
            return x_prev
            
        if abs(fx_curr) < best_abs_fx:
            best_x, best_fx, best_abs_fx = x_curr, fx_curr, abs(fx_curr)
        if abs(fx_prev) < best_abs_fx:
            best_x, best_fx, best_abs_fx = x_prev, fx_prev, abs(fx_prev)

        # Check for sign change between current two points -> Brent
        if fx_prev * fx_curr < 0:
            try:
                root = brentq(f, x_prev, x_curr, maxiter=50, rtol=1e-12, xtol=1e-14)
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_abs_fx:
                    return root
            except Exception:
                pass
            # If brentq failed, fall through to secant step? Or break?
            # Secant step is safer to continue if brentq fails due to weird float issues.
            
        # Secant step
        if fx_curr == fx_prev:
            # Parallel lines, no root estimate. Perturb.
            x_new = x_curr + 1e-5 * max(1.0, abs(x_curr))
        else:
            # x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            denom = fx_curr - fx_prev
            if denom == 0:
                x_new = x_curr + 1e-5 * max(1.0, abs(x_curr))
            else:
                x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
                
        # Sanity check on step size
        # If step is huge, damp it to prevent floating point overflow or jumping to infinity
        max_step = 10.0 * (1.0 + abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            # Damp the step
            direction = 1.0 if x_new > x_curr else -1.0
            x_new = x_curr + direction * max_step

        fx_new = safe_eval(x_new)
        
        if fx_new is None:
            # Evaluation failed (inf/nan). 
            # This usually means we stepped into a singularity or domain error.
            # Try to recover by shrinking step or using bisection if bracket exists.
            # For now, if we can't evaluate, we are stuck.
            # Try a smaller step?
            x_try = x_curr - 0.5 * (x_new - x_curr)
            fx_try = safe_eval(x_try)
            if fx_try is None:
                break
            fx_new = fx_try
            x_new = x_try
            
        # Update points
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
        
    return best_x
