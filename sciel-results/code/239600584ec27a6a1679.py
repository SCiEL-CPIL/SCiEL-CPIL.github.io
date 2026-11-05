"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0  # Safety margin for 150s limit
    
    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, FloatingPointError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val

    def is_good(x, fx):
        if fx is None:
            return False
        return abs(fx) <= TOL

    # --- Trust Region Secant ---
    # This method is structurally different from simple Secant/Brent.
    # It maintains a trust region [x - delta, x + delta] and only accepts
    # steps that stay within it, shrinking the region on failure.
    # This provides robustness against the "runaway" behavior of standard
    # secant methods on ill-conditioned problems.
    
    def trust_region_secant(x0):
        nonlocal evals
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return x, None
        if is_good(x, fx):
            return x, fx
        
        # Initial delta based on scale
        delta = max(1e-2, 1e-2 * abs(x))
        
        # We need a second point to start secant.
        # Try a point slightly offset.
        x1 = x + delta
        fx1 = call_f(x1)
        
        if fx1 is None:
             # If we can't evaluate x1, try x - delta
             x1 = x - delta
             fx1 = call_f(x1)
             if fx1 is None:
                 return x, fx
             
        if is_good(x1, fx1):
            return x1, fx1

        # We have (x, fx) and (x1, fx1).
        # Trust region is centered at x, radius delta.
        # Actually, standard Trust Region Secant (e.g., in MINPACK) uses:
        # x_new = x - f(x) * (x - x1) / (f(x) - f(x1))
        # If |x_new - x| > delta, clamp x_new to x +/- delta.
        # If f(x_new) < f(x), accept and update.
        # If f(x_new) > f(x), reject and shrink delta.
        
        # To make it more robust, we can alternate or just stick to the secant step
        # constrained by the trust region.
        
        # Let's refine the initial state.
        # We want to find a root.
        # If f(x) and f(x1) have opposite signs, we could also switch to bisection/secant hybrid.
        # But Trust Region Secant works well even without sign change if it minimizes |f|.
        
        # However, for root finding, if we don't have a sign change, we might be minimizing.
        # The task says "ill-conditioned", which often implies steep gradients or near-flat regions.
        
        # Let's implement a robust version:
        # 1. Try Secant step.
        # 2. If step is too large, clamp.
        # 3. Evaluate.
        # 4. If |f| decreases, move.
        # 5. If |f| increases, shrink trust region and try again.
        
        x_curr = x
        fx_curr = fx
        x_prev = x1
        fx_prev = fx1
        
        # Ensure x_curr is the "better" point (smaller |f|) to drive minimization?
        # No, Secant uses the line between them.
        
        # Let's run the loop
        for _ in range(100):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                break
                
            if fx_curr == fx_prev:
                # Flat function or numerical issue
                break
                
            # Secant step
            x_step = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            
            # Trust region constraint
            # The trust region is centered at x_curr? Or fixed?
            # Standard TRS: Region is around current point.
            # If x_step is outside [x_curr - delta, x_curr + delta], move to boundary.
            max_step = delta
            if x_step > x_curr + max_step:
                x_step = x_curr + max_step
            elif x_step < x_curr - max_step:
                x_step = x_curr - max_step
                
            if not math.isfinite(x_step):
                break
                
            fx_step = call_f(x_step)
            if fx_step is None:
                break
                
            if is_good(x_step, fx_step):
                return x_step, fx_step
                
            # Decision: Accept or Reject
            # For root finding, we want |f| to decrease.
            if abs(fx_step) < abs(fx_curr):
                # Accept
                # Update history
                x_prev = x_curr
                fx_prev = fx_curr
                x_curr = x_step
                fx_curr = fx_step
                # Expand trust region slightly if we moved a lot? Or keep it?
                # Usually, if we move to the boundary, we might want to expand.
                # If we moved inside, we can keep or expand.
                # Let's expand by 2 if we hit the boundary, otherwise keep.
                if abs(x_step - x_curr) >= max_step * 0.99: # We are at boundary? No, x_curr is old.
                    # We moved to boundary
                    delta = min(delta * 2.0, 10.0) # Cap expansion
                else:
                    # We moved inside, maybe shrink slightly to focus?
                    # Or keep. Let's keep.
                    pass
            else:
                # Reject
                # Shrink trust region
                delta = max(delta * 0.5, 1e-15)
                # Do not update x_curr or x_prev.
                # Try again with smaller delta.
                # To avoid infinite loop on same step, we might need to perturb x_prev?
                # If we keep rejecting, we are stuck.
                # If delta is very small, we might be at a local minimum of |f| or a root.
                if delta < 1e-14:
                    break
        
        return x_curr, fx_curr

    # --- Golden Section Minimization ---
    # Fallback for cases where no sign change is found and Trust Region Secant
    # fails to converge (e.g., tangential roots where f touches 0 but doesn't cross).
    
    def golden_section_min(x0, width):
        nonlocal evals
        # Minimize |f(x)| in [x0 - width, x0 + width]
        a = x0 - width
        b = x0 + width
        
        fa = call_f(a)
        fb = call_f(b)
        
        if fa is None or fb is None:
            # Try to recover
            if fa is None and fb is not None:
                return b, fb
            if fb is None and fa is not None:
                return a, fa
            return x0, None
            
        if is_good(a, fa):
            return a, fa
        if is_good(b, fb):
            return b, fb
        
        # Golden ratio
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = call_f(c)
        fd = call_f(d)
        
        if fc is None or fd is None:
             # Return best
             best_x_local = a
             best_f_local = fa
             if fb is not None and abs(fb) < abs(fa):
                 best_x_local = b
                 best_f_local = fb
             if fc is not None and abs(fc) < abs(best_f_local):
                 best_x_local = c
                 best_f_local = fc
             if fd is not None and abs(fd) < abs(best_f_local):
                 best_x_local = d
                 best_f_local = fd
             return best_x_local, best_f_local

        if is_good(c, fc):
            return c, fc
        if is_good(d, fd):
            return d, fd
        
        for _ in range(50):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                break
                
            if abs(fc) < abs(fd):
                b = d
                fb = fd
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = call_f(c)
                if fc is None:
                    break
            else:
                a = c
                fa = fc
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = call_f(d)
                if fd is None:
                    break
                    
            if is_good(c, fc):
                return c, fc
            if is_good(d, fd):
                return d, fd
            
            # Convergence check
            if abs(b - a) < 1e-12 * max(1.0, abs(x0)):
                break
                
        # Return best of c, d
        if abs(fc) < abs(fd):
            return c, fc
        else:
            return d, fd

    # --- Strategy Execution ---
    
    # 1. Trust Region Secant (Primary)
    # This is robust for both crossing and tangential roots if it minimizes |f|.
    x_res, f_res = trust_region_secant(x0)
    if is_good(x_res, f_res):
        return x_res
        
    # 2. Golden Section Minimization (Secondary)
    # If TRS failed, we might be in a region where the function is flat or we lost the root.
    # Minimize |f| in a reasonable window around x0.
    # Start with a small window, expand if necessary?
    # For now, try a few windows.
    
    # Window 1: Small
    x_res, f_res = golden_section_min(x0, 1e-2)
    if is_good(x_res, f_res):
        return x_res
        
    # Window 2: Medium
    x_res, f_res = golden_section_min(x0, 1.0)
    if is_good(x_res, f_res):
        return x_res
        
    # Window 3: Large
    x_res, f_res = golden_section_min(x0, 10.0)
    if is_good(x_res, f_res):
        return x_res

    # 3. Final Polish
    # If we have a good candidate from best_x, try to polish it with a tiny TRS or Newton.
    if best_abs < 1e-4:
        # Use a very small trust region around best_x
        x_res, f_res = trust_region_secant(best_x)
        if is_good(x_res, f_res):
            return x_res
        # Or just return best_x if it's close enough? No, must be <= 1e-8.
        # If best_abs is close to 1e-8, we might need to squeeze it.
        # Try a few random perturbations? No, deterministic is better.
        
    # If all fail, return best_x found so far
    return best_x

# EVOLVE-BLOCK-END
