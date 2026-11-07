"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 110.0  # 110s safety margin before 120s kill

    def time_up():
        return time.time() > deadline

    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)

    if best_abs <= 1e-8:
        return x

    # Phase 1: Inverse interpolation with Regula Falsi safeguard
    # Start with a small perturbation to get two points
    h = 1e-4 * max(1.0, abs(x))
    x_a = x
    x_b = x + h
    fa = best_fx
    fb = f(x_b)

    if abs(fb) < best_abs:
        best_x, best_fx, best_abs = x_b, fb, abs(fb)
        if best_abs <= 1e-8:
            return x_b

    # Try to establish a bracket for Regula Falsi safeguard
    # If no sign change, expand to find one
    a, b, fa, fb = x_a, x_b, fa, fb
    has_bracket = math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0

    if not has_bracket:
        # Expand to find a bracket
        step = 1.0
        for _ in range(40):
            if time_up():
                break
            a_new = best_x - step
            b_new = best_x + step
            fa_new = f(a_new)
            fb_new = f(b_new)
            if not (math.isfinite(fa_new) and math.isfinite(fb_new)):
                step *= 2.0
                continue
            if fa_new * fb_new < 0:
                a, b, fa, fb = a_new, b_new, fa_new, fb_new
                has_bracket = True
                break
            step *= 2.0
            if step > 1e10:
                break

    # Main loop: inverse quadratic/secant with Regula Falsi safeguard
    # We maintain two points (x1, f1) and (x2, f2) and try inverse interpolation
    # If no bracket, use secant; if bracket exists, use Regula Falsi-like step
    # but with inverse interpolation when possible
    
    # Initialize the two working points
    # Choose the two points with smallest |f|
    if abs(fa) <= abs(fb):
        x1, f1, x2, f2 = x_a, fa, x_b, fb
    else:
        x1, f1, x2, f2 = x_b, fb, x_a, fa

    for i in range(50):
        if time_up():
            break

        # Check convergence
        if abs(f2) <= 1e-8:
            return x2
        if abs(f1) <= 1e-8:
            return x1

        # Compute inverse interpolation (secant) step
        if f1 != f2 and abs(f2 - f1) > 1e-30:
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            break

        # Check if x_new is reasonable
        if not math.isfinite(x_new):
            break

        # If we have a bracket, ensure x_new stays within it (Regula Falsi safeguard)
        if has_bracket:
            # Determine which endpoint has smaller |f|
            if abs(fa) < abs(fb):
                # fa is smaller, x_new should be between x_b and x_new_secant
                # Actually, let's just check if x_new is in [min(a,b), max(a,b)]
                lo = min(a, b)
                hi = max(a, b)
                if x_new < lo or x_new > hi:
                    # Fall back to bisection or regula falsi
                    # Use the standard regula falsi step
                    if fa * fb != 0:
                        x_new = (a * fb - b * fa) / (fb - fa)
                    else:
                        x_new = 0.5 * (a + b)
                    if not math.isfinite(x_new):
                        break
                    lo = min(a, b)
                    hi = max(a, b)
                    if x_new < lo or x_new > hi:
                        x_new = 0.5 * (a + b)
            else:
                lo = min(a, b)
                hi = max(a, b)
                if x_new < lo or x_new > hi:
                    if fa * fb != 0:
                        x_new = (a * fb - b * fa) / (fb - fa)
                    else:
                        x_new = 0.5 * (a + b)
                    if not math.isfinite(x_new):
                        break
                    lo = min(a, b)
                    hi = max(a, b)
                    if x_new < lo or x_new > hi:
                        x_new = 0.5 * (a + b)

        f_new = f(x_new)
        if not math.isfinite(f_new):
            break

        if abs(f_new) < best_abs:
            best_x, best_fx, best_abs = x_new, f_new, abs(f_new)

        if best_abs <= 1e-8:
            return x_new

        # Update the two working points
        # Keep the two points with smallest |f|
        if abs(f_new) < abs(f1):
            # Replace x1 with x_new
            x1, f1 = x_new, f_new
        elif abs(f_new) < abs(f2):
            # Replace x2 with x_new
            x2, f2 = x_new, f_new
        else:
            # f_new is larger than both, discard it but update bracket if applicable
            pass

        # Update bracket if we have one
        if has_bracket:
            if fa * f_new <= 0:
                # New bracket is (a, x_new) or (x_new, b) depending on signs
                if fa * f_new < 0:
                    # x_new and a have opposite signs
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
            elif fb * f_new <= 0:
                if fb * f_new < 0:
                    # x_new and b have opposite signs
                    a, fa = x_new, f_new
                else:
                    b, fb = x_new, f_new
            # If f_new has same sign as both, bracket unchanged

    # Phase 2: If we still have a bracket, use Brent's method for final polish
    if has_bracket and not time_up():
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # Phase 3: Final bisection if we have a bracket
    if has_bracket and not time_up():
        for _ in range(60):
            if time_up():
                break
            mid = 0.5 * (a + b)
            fm = f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) < best_abs:
                best_x, best_fx, best_abs = mid, fm, abs(fm)
            if best_abs <= 1e-8:
                return mid
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm

    return best_x
# EVOLVE-BLOCK-END
