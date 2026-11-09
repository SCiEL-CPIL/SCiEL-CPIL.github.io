"""Root finder using Steffensen's method with adaptive step sizing."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, starting near x0.
    Uses Steffensen's method: x_{n+1} = x_n - f(x_n)^2 / (f(x_n + f(x_n)) - f(x_n))
    This is equivalent to Aitken's delta-squared acceleration of fixed-point iteration.
    Requires 3 evaluations per step, no derivative, no bracketing.
    """
    deadline = time.time() + 145.0
    tol = 1e-9  # tighter than 1e-8 for safety
    
    x = float(x0)
    best_x = x
    best_abs = float('inf')
    
    # Initial evaluation
    try:
        fx = f(x)
        if fx is not None and math.isfinite(fx):
            best_abs = abs(fx)
            if best_abs <= tol:
                return x
    except Exception:
        pass
    
    # Try Steffensen's method with adaptive perturbation
    # The key idea: if f(x) is small, use a small perturbation; if large, use larger
    # Steffensen: x_new = x - f(x)^2 / (f(x + f(x)) - f(x))
    # But we adapt: use delta = f(x) * scale_factor, where scale_factor adapts
    
    # Phase 1: Pure Steffensen iterations
    x_curr = x
    for i in range(200):
        if time.time() > deadline - 5.0:
            break
        
        try:
            fx = f(x_curr)
        except Exception:
            break
        if fx is None or not math.isfinite(fx):
            break
        
        if abs(fx) < best_abs:
            best_abs = abs(fx)
            best_x = x_curr
        if best_abs <= tol:
            return best_x
        
        # Adaptive perturbation: proportional to |f(x)| but bounded
        # If f(x) is very small, we're close to a root; use tiny step
        # If f(x) is large, use a step that's a fraction of f(x)
        abs_fx = abs(fx)
        
        if abs_fx < 1e-12:
            # Essentially at a root, but not quite precise enough
            # Try a tiny nudge
            delta = 1e-12 * math.copysign(1.0, fx if fx != 0 else 1.0)
        elif abs_fx < 1e-6:
            delta = abs_fx * 1.0
        elif abs_fx < 1e-2:
            delta = abs_fx * 0.5
        else:
            # For large f(x), limit the step to avoid flying off
            delta = abs_fx * 0.1
            # Cap the absolute step size
            max_step = 1.0 * max(1.0, abs(x_curr))
            if abs(delta) > max_step:
                delta = math.copysign(max_step, fx)
        
        x_plus = x_curr + delta
        
        try:
            f_plus = f(x_plus)
        except Exception:
            break
        if f_plus is None or not math.isfinite(f_plus):
            # Try opposite direction
            x_plus = x_curr - delta
            try:
                f_plus = f(x_plus)
            except Exception:
                break
            if f_plus is None or not math.isfinite(f_plus):
                break
        
        # Steffensen's formula: x_new = x - f(x)^2 / (f(x+delta) - f(x))
        # But with adaptive delta, we use: x_new = x - f(x) * delta / (f(x+delta) - f(x))
        # This is the secant-like form that generalizes Steffensen
        denom = f_plus - fx
        if abs(denom) < 1e-30:
            break
        
        x_new = x_curr - fx * delta / denom
        
        if not math.isfinite(x_new):
            break
        
        # Safety: if step is too large, damp it
        max_move = 5.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_move:
            x_new = x_curr + math.copysign(max_move, x_new - x_curr)
        
        x_curr = x_new
    
    if best_abs <= tol:
        return best_x
    
    # Phase 2: If Steffensen didn't converge, try a local search with geometric expansion
    # to find a sign change, then bisection
    result = _local_bracket_and_bisect(f, x, best_x, best_abs, tol, deadline)
    
    # Phase 3: Polish with a few more Steffensen steps from the best candidate
    if best_abs <= 1e-5 and best_abs > tol:
        polished = _steffensen_polish(f, best_x, best_abs, tol, deadline)
        if polished is not None:
            return polished
    
    return result


def _local_bracket_and_bisect(f, x0, best_x, best_abs, tol, deadline):
    """Geometric expansion to find a sign change, then bisection."""
    scale = 0.1 * max(1.0, abs(x0))
    
    for attempt in range(15):
        if time.time() > deadline - 3.0:
            break
        
        lo = x0 - scale
        hi = x0 + scale
        
        try:
            flo = f(lo)
        except Exception:
            scale *= 2.0
            continue
        if flo is None or not math.isfinite(flo):
            scale *= 2.0
            continue
        
        try:
            fhi = f(hi)
        except Exception:
            scale *= 2.0
            continue
        if fhi is None or not math.isfinite(fhi):
            scale *= 2.0
            continue
        
        if abs(flo) < best_abs:
            best_abs = abs(flo)
            best_x = lo
        if abs(fhi) < best_abs:
            best_abs = abs(fhi)
            best_x = hi
        if best_abs <= tol:
            return best_x
        
        if flo * fhi < 0:
            # Found a bracket, bisect
            return _bisect(f, lo, hi, flo, fhi, best_x, best_abs, tol, deadline)
        
        scale *= 2.0
    
    return best_x


def _bisect(f, lo, hi, flo, fhi, best_x, best_abs, tol, deadline):
    """Standard bisection."""
    for i in range(60):
        if time.time() > deadline - 2.0:
            break
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        try:
            fmid = f(mid)
        except Exception:
            break
        if fmid is None or not math.isfinite(fmid):
            break
        if abs(fmid) < best_abs:
            best_abs = abs(fmid)
            best_x = mid
        if best_abs <= tol:
            return best_x
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return best_x


def _steffensen_polish(f, x, best_abs, tol, deadline):
    """A few Steffensen steps from a good candidate to achieve tight tolerance."""
    x = float(x)
    for i in range(20):
        if time.time() > deadline - 1.0:
            return None
        try:
            fx = f(x)
        except Exception:
            return None
        if fx is None or not math.isfinite(fx):
            return None
        if abs(fx) <= tol:
            return x
        if abs(fx) > best_abs * 10:
            return None
        
        # Small adaptive step for polish
        abs_fx = abs(fx)
        if abs_fx < 1e-14:
            delta = 1e-14 * math.copysign(1.0, fx if fx != 0 else 1.0)
        elif abs_fx < 1e-10:
            delta = abs_fx
        else:
            delta = abs_fx * 0.5
        
        x_plus = x + delta
        try:
            f_plus = f(x_plus)
        except Exception:
            return None
        if f_plus is None or not math.isfinite(f_plus):
            return None
        
        denom = f_plus - fx
        if abs(denom) < 1e-30:
            return None
        
        x_new = x - fx * delta / denom
        if not math.isfinite(x_new):
            return None
        
        # Very conservative step for polish
        max_move = 1e-6 * max(1.0, abs(x))
        if abs(x_new - x) > max_move:
            x_new = x + math.copysign(max_move, x_new - x)
        
        x = x_new
    
    return None
# EVOLVE-BLOCK-END
