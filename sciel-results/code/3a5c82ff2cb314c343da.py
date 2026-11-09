"""Root finder for smooth polynomial equations with call-count optimization."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for a smooth polynomial.
    
    Strategy:
    1. Evaluate f(x0). If |f(x0)| <= TOL, return immediately.
    2. Try secant method with a small perturbation. If it converges quickly, return.
    3. If secant fails, switch to bracketing + Illinois + secant hybrid.
    4. Final polish pass with tighter tolerance if close.
    """
    deadline = time.time() + 115.0
    TOL = 1e-8
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        for dx in [1e-4, -1e-4, 0.1, -0.1, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except Exception:
                continue
        else:
            return x
    
    if not math.isfinite(fx):
        for dx in [1e-4, -1e-4, 0.1, -0.1, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except Exception:
                continue
        else:
            return x
    
    best_x = x
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= TOL:
        return x
    
    # Phase 1: Try secant method with a small perturbation
    # Use a single well-chosen step to create two initial points
    step = 1e-4 * max(1.0, abs(x))
    x_prev = x - step
    x_curr = x
    f_curr = fx
    
    try:
        f_prev = f(x_prev)
    except Exception:
        f_prev = None
    
    if f_prev is None or not math.isfinite(f_prev):
        # Try a few other perturbations
        for s in [1e-3, -1e-3, 0.1, -0.1]:
            x_try = x + s * max(1.0, abs(x))
            try:
                f_try = f(x_try)
                if math.isfinite(f_try):
                    x_prev = x_try
                    f_prev = f_try
                    break
            except Exception:
                continue
    
    if f_prev is None or not math.isfinite(f_prev):
        # Cannot do secant, go directly to bracketing
        return _bracket_and_solve(f, x, fx, best_x, best_abs_fx, deadline, TOL)
    
    # Secant iteration with early termination
    for i in range(20):
        if time.time() > deadline:
            break
        
        if abs(f_curr) <= TOL:
            best_x = x_curr
            best_abs_fx = abs(f_curr)
            return _polish(f, x_curr, f_curr, TOL, deadline)
        if abs(f_prev) <= TOL:
            best_x = x_prev
            best_abs_fx = abs(f_prev)
            return _polish(f, x_prev, f_prev, TOL, deadline)
        
        if abs(f_curr) < best_abs_fx:
            best_abs_fx = abs(f_curr)
            best_x = x_curr
        if abs(f_prev) < best_abs_fx:
            best_abs_fx = abs(f_prev)
            best_x = x_prev
        
        if f_curr == f_prev:
            break
        
        denom = f_curr - f_prev
        if abs(denom) < 1e-30:
            break
        
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            break
        
        # Dampen if step is too large
        max_step = 5.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + 0.5 * (x_new - x_curr) * (1.0 if x_new > x_curr else -1.0)
        
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        
        try:
            f_curr = f(x_curr)
        except Exception:
            break
        
        if not math.isfinite(f_curr):
            # Revert and go to bracketing
            x_curr, f_curr = x_prev, f_prev
            return _bracket_and_solve(f, x, fx, best_x, best_abs_fx, deadline, TOL)
    
    # If secant converged to tolerance, return
    if abs(f_curr) <= TOL:
        return _polish(f, x_curr, f_curr, TOL, deadline)
    if abs(f_prev) <= TOL:
        return _polish(f, x_prev, f_prev, TOL, deadline)
    
    # Phase 2: Secant didn't converge. Use bracketing + Illinois + secant
    return _bracket_and_solve(f, x, fx, best_x, best_abs_fx, deadline, TOL)


def _polish(f, x, fx, TOL, deadline):
    """Final polish pass: if very close, do a few more secant steps with tighter tolerance."""
    if abs(fx) <= TOL:
        return x
    
    # If we're within 1e-6, try a few more secant steps to get tighter
    if abs(fx) > 1e-6:
        return x
    
    # Try secant with a tiny perturbation for final polish
    step = 1e-6 * max(1.0, abs(x))
    x_prev = x - step
    try:
        f_prev = f(x_prev)
    except Exception:
        return x
    
    if not math.isfinite(f_prev):
        return x
    
    x_curr = x
    f_curr = fx
    
    for i in range(6):
        if time.time() > deadline:
            break
        
        if abs(f_curr) <= TOL:
            return x_curr
        if abs(f_prev) <= TOL:
            return x_prev
        
        denom = f_curr - f_prev
        if abs(denom) < 1e-30:
            break
        
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            break
        
        # Keep steps small during polish
        max_step = 1e-3 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + 0.5 * (x_new - x_curr) * (1.0 if x_new > x_curr else -1.0)
        
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        
        try:
            f_curr = f(x_curr)
        except Exception:
            break
        
        if not math.isfinite(f_curr):
            break
    
    # Return the best point found
    if abs(f_curr) <= abs(f_prev):
        return x_curr
    else:
        return x_prev


def _bracket_and_solve(f, x_start, fx_start, best_x, best_abs_fx, deadline, TOL):
    """Find a bracket and solve using Illinois method, then secant for final convergence."""
    x = x_start
    fx = fx_start
    
    # Try to find a bracket by expanding from x
    a = x
    fa = fx
    bracketed = False
    b = None
    fb = None
    
    # Try expanding in both directions with geometric growth
    for scale in [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
        
        # Try positive direction
        x_plus = a + scale * max(1.0, abs(a))
        try:
            f_plus = f(x_plus)
            if math.isfinite(f_plus):
                if fa * f_plus < 0:
                    bracketed = True
                    b = x_plus
                    fb = f_plus
                    break
        except Exception:
            pass
        
        # Try negative direction
        x_minus = a - scale * max(1.0, abs(a))
        try:
            f_minus = f(x_minus)
            if math.isfinite(f_minus):
                if fa * f_minus < 0:
                    bracketed = True
                    b = x_minus
                    fb = f_minus
                    break
        except Exception:
            pass
    
    if not bracketed:
        # No bracket found. Fall back to pure secant from the best point
        step = 1e-2 * max(1.0, abs(x))
        x_prev = x - step
        x_curr = x
        f_curr = fx
        
        try:
            f_prev = f(x_prev)
        except Exception:
            f_prev = fx + 1e-4
        
        if not math.isfinite(f_prev):
            f_prev = fx + 1e-4
        
        for i in range(25):
            if time.time() > deadline:
                break
            
            if abs(f_curr) <= TOL:
                return _polish(f, x_curr, f_curr, TOL, deadline)
            if abs(f_prev) <= TOL:
                return _polish(f, x_prev, f_prev, TOL, deadline)
            
            if abs(f_curr) < best_abs_fx:
                best_abs_fx = abs(f_curr)
                best_x = x_curr
            if abs(f_prev) < best_abs_fx:
                best_abs_fx = abs(f_prev)
                best_x = x_prev
            
            if f_curr == f_prev:
                break
            
            denom = f_curr - f_prev
            if abs(denom) < 1e-30:
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            
            if not math.isfinite(x_new):
                break
            
            max_step = 5.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr + 0.5 * (x_new - x_curr) * (1.0 if x_new > x_curr else -1.0)
            
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
            
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            
            if not math.isfinite(f_curr):
                break
        
        return best_x
    
    # We have a bracket [a, b] with fa * fb < 0
    # Use Illinois method to narrow the bracket
    # Ensure a has smaller |f|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        if abs(fa) <= TOL:
            return _polish(f, a, fa, TOL, deadline)
        if abs(fb) <= TOL:
            return _polish(f, b, fb, TOL, deadline)
        
        if fa * fb > 0:
            break
        
        # Illinois step: use secant-like step but with damping
        denom = fb - fa
        if abs(denom) < 1e-30:
            break
        
        c = b - fb * (b - a) / denom
        fc = None
        try:
            fc = f(c)
        except Exception:
            c = 0.5 * (a + b)
            try:
                fc = f(c)
            except Exception:
                break
        
        if not math.isfinite(fc):
            c = 0.5 * (a + b)
            try:
                fc = f(c)
            except Exception:
                break
        
        if abs(fc) <= TOL:
            return _polish(f, c, fc, TOL, deadline)
        
        if abs(fc) < best_abs_fx:
            best_abs_fx = abs(fc)
            best_x = c
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = 0.5 * fa
            fb = fc
    
    # After bracketing, if we haven't reached TOL, do a few secant steps
    # from the last two points
    x1, f1 = a, fa
    x2, f2 = b, fb
    
    for i in range(12):
        if time.time() > deadline:
            break
        
        if abs(f1) <= TOL:
            return _polish(f, x1, f1, TOL, deadline)
        if abs(f2) <= TOL:
            return _polish(f, x2, f2, TOL, deadline)
        
        if abs(f1) < best_abs_fx:
            best_abs_fx = abs(f1)
            best_x = x1
        if abs(f2) < best_abs_fx:
            best_abs_fx = abs(f2)
            best_x = x2
        
        if f1 == f2:
            break
        
        denom = f2 - f1
        if abs(denom) < 1e-30:
            break
        
        x_new = x1 - f1 * (x2 - x1) / denom
        
        if not math.isfinite(x_new):
            break
        
        max_step = 5.0 * max(1.0, abs(x1))
        if abs(x_new - x1) > max_step:
            x_new = x1 + 0.5 * (x_new - x1) * (1.0 if x_new > x1 else -1.0)
        
        x1, f1 = x2, f2
        try:
            f_new = f(x_new)
        except Exception:
            break
        
        if not math.isfinite(f_new):
            break
        
        x2 = x_new
        f2 = f_new
    
    return best_x

# EVOLVE-BLOCK-END
