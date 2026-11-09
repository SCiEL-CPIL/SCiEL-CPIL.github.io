"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _safe_f(f, x):
    try:
        y = f(x)
        if not math.isfinite(y):
            return float('inf')
        return float(y)
    except Exception:
        return float('inf')

def find_root(f, x0):
    """
    Find a root x such that |f(x)| <= 1e-8.
    Strategy:
    1. Evaluate f at x0. If close enough, return x0.
    2. Try a short secant/Newton hybrid from x0 to quickly converge if x0 is near a root.
    3. If not close, search for a bracket [a, b] where f(a)*f(b) < 0 by expanding the window.
       Once a bracket is found, use Illinois/modified bisection to converge.
    4. Return the best point found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    # Phase 1: Quick Newton/Secant attempt from x0
    # This uses a finite-difference derivative and falls back to bisection-like steps
    # if Newton diverges. We limit this to ~15 iterations.
    x_a = x0
    fa = _safe_f(f, x_a)

    if math.isfinite(fa) and abs(fa) <= 1e-8:
        return x_a

    # Try secant with x1 = x0 + h
    h0 = 1e-4 * max(1.0, abs(x0))
    x_b = x_a + h0
    fb = _safe_f(f, x_b)

    if math.isfinite(fb) and abs(fb) <= 1e-8:
        return x_b

    best_x = x_a
    best_abs = abs(fa) if math.isfinite(fa) else float('inf')

    # Secant/Newton hybrid: use the two points, and if secant step goes outside
    # the bracket or diverges, use a bisection step instead.
    # We maintain a bracket if possible, otherwise just track best point.
    
    if math.isfinite(fb):
        # Track best
        if abs(fb) < best_abs:
            best_abs = abs(fb)
            best_x = x_b

        # Try up to 15 secant/bisection steps
        # We'll try to maintain a bracket [lo, hi] if we have a sign change.
        lo, hi = None, None
        flo, fhi = None, None
        
        if fa * fb < 0:
            # We have a bracket
            if abs(fa) < abs(fb):
                lo, hi = x_a, x_b
                flo, fhi = fa, fb
            else:
                lo, hi = x_b, x_a
                flo, fhi = fb, fa

        for _ in range(15):
            if time.time() > deadline - 20.0:
                break
                
            # If we have a bracket, use Illinois method (modified bisection)
            if lo is not None and hi is not None:
                # Illinois: use bisection but weighted toward the side with smaller |f|
                if abs(flo) < abs(fhi):
                    # Use secant-like step from hi, but constrained to bracket
                    if abs(fhi - flo) > 1e-15:
                        x_new = hi - fhi * (hi - lo) / (fhi - flo)
                    else:
                        x_new = (lo + hi) / 2.0
                    # Constrain to bracket
                    if x_new <= lo or x_new >= hi:
                        x_new = (lo + hi) / 2.0
                else:
                    if abs(fhi - flo) > 1e-15:
                        x_new = lo - flo * (hi - lo) / (fhi - flo)
                    else:
                        x_new = (lo + hi) / 2.0
                    if x_new <= lo or x_new >= hi:
                        x_new = (lo + hi) / 2.0
                    
                f_new = _safe_f(f, x_new)
                if not math.isfinite(f_new):
                    # Fallback to pure bisection
                    x_new = (lo + hi) / 2.0
                    f_new = _safe_f(f, x_new)
                    if not math.isfinite(f_new):
                        break
                
                if abs(f_new) < best_abs:
                    best_abs = abs(f_new)
                    best_x = x_new
                    
                if abs(f_new) <= 1e-8:
                    return x_new

                # Update bracket
                if flo * f_new <= 0:
                    hi = x_new
                    fhi = f_new
                else:
                    lo = x_new
                    flo = f_new
                    # Illinois: dampen the flo side
                    if abs(flo) > abs(fhi):
                        pass # standard update
                # Actually, let's just do standard bisection for reliability in the hybrid
                # The secant step above is an enhancement; let's simplify to bisection for safety
                # and just use secant when no bracket.
                break  # Switch to dedicated bracket search phase if this gets complicated

            # No bracket: try secant step
            if abs(fb - fa) > 1e-15:
                x_new = x_a - fa * (x_a - x_b) / (fa - fb)
            else:
                break

            if not math.isfinite(x_new):
                break

            # Check if x_new is reasonable (not too far)
            if abs(x_new - x0) > 100 * max(1.0, abs(x0)) + 100:
                # Too far, try a smaller step
                x_new = x_a + 0.5 * (x_b - x_a)
            
            f_new = _safe_f(f, x_new)
            if not math.isfinite(f_new):
                break

            if abs(f_new) < best_abs:
                best_abs = abs(f_new)
                best_x = x_new

            if abs(f_new) <= 1e-8:
                return x_new

            # Check for new bracket
            if fa * f_new < 0:
                # Update bracket
                if abs(fa) < abs(f_new):
                    lo, hi = x_a, x_new
                    flo, fhi = fa, f_new
                else:
                    lo, hi = x_new, x_a
                    flo, fhi = f_new, fa
                # Continue with bracket phase
                continue
            elif fb * f_new < 0:
                if abs(fb) < abs(f_new):
                    lo, hi = x_b, x_new
                    flo, fhi = fb, f_new
                else:
                    lo, hi = x_new, x_b
                    flo, fhi = f_new, fb
                continue

            # No bracket, update secant points
            if abs(f_new) < abs(fb):
                x_a, x_b = x_b, x_new
                fa, fb = fb, f_new
            else:
                x_a, x_b = x_a, x_new
                fa, fb = fa, f_new

    # Phase 2: Dedicated bracket search with expanding window
    # Start from x0 and expand in both directions
    a = x0
    fa = _safe_f(f, a)
    if not math.isfinite(fa):
        a = 0.0
        fa = _safe_f(f, a)
        if not math.isfinite(fa):
            return best_x

    if abs(fa) < best_abs:
        best_abs = abs(fa)
        best_x = a

    if abs(fa) <= 1e-8:
        return a

    scale = 1.0
    found_bracket = False
    b_val, fb_val = None, None

    while time.time() < deadline - 10.0 and scale < 1e10:
        for direction in [1.0, -1.0]:
            b = a + direction * scale
            fb = _safe_f(f, b)
            if not math.isfinite(fb):
                continue

            if abs(fb) < best_abs:
                best_abs = abs(fb)
                best_x = b

            if abs(fb) <= 1e-8:
                return b

            if fa * fb < 0:
                found_bracket = True
                # We have a bracket [a, b]
                # Use Illinois method to converge
                lo, hi = a, b
                flo, fhi = fa, fb

                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    # Illinois method
                    if abs(flo) < abs(fhi):
                        # Use secant step from hi
                        if abs(fhi - flo) > 1e-15:
                            m = hi - fhi * (hi - lo) / (fhi - flo)
                        else:
                            m = (lo + hi) / 2.0
                    else:
                        if abs(fhi - flo) > 1e-15:
                            m = lo - flo * (hi - lo) / (fhi - flo)
                        else:
                            m = (lo + hi) / 2.0

                    # Constrain to bracket
                    if m <= lo or m >= hi:
                        m = (lo + hi) / 2.0

                    fm = _safe_f(f, m)
                    if not math.isfinite(fm):
                        # Fallback to bisection
                        m = (lo + hi) / 2.0
                        fm = _safe_f(f, m)
                        if not math.isfinite(fm):
                            break

                    if abs(fm) < best_abs:
                        best_abs = abs(fm)
                        best_x = m

                    if abs(fm) <= 1e-8:
                        return m

                    if flo * fm <= 0:
                        hi = m
                        fhi = fm
                    else:
                        lo = m
                        flo = fm

                    if abs(hi - lo) < 1e-12:
                        break

                return (lo + hi) / 2.0

        scale *= 2.0

    # If no bracket found, return best_x
    return best_x

# EVOLVE-BLOCK-END
