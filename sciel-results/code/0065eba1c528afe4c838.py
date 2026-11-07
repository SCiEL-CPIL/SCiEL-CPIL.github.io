"""Root finder for 3 fixed smooth polynomial equations."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0

    x = float(x0)

    # Deadline check helper
    def _deadline_exceeded():
        return time.time() > deadline - 2.0

    # ---- Phase 1: Adaptive cubic interpolation + Newton polish ----
    # Evaluate f at 4 points around x0 to build a cubic model
    # Use a symmetric stencil for better conditioning
    h = 1.0
    stencil = [x - 1.5*h, x - 0.5*h, x + 0.5*h, x + 1.5*h]

    try:
        vals = []
        for p in stencil:
            if _deadline_exceeded():
                break
            vals.append(f(p))

        if len(vals) == 4:
            # Fit cubic polynomial
            coeffs = np.polyfit(stencil, vals, 3)
            roots = np.roots(coeffs)

            # Collect real roots
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]

            if real_roots:
                # Pick the real root closest to x0
                candidate = min(real_roots, key=lambda r: abs(r - x))

                # Verify
                fc = f(candidate)
                if abs(fc) <= 1e-8:
                    return candidate

                # If not good enough, try Newton from the candidate
                xn = candidate
                for _ in range(30):
                    if _deadline_exceeded():
                        break
                    fxn = f(xn)
                    if abs(fxn) <= 1e-8:
                        return xn
                    # Numerical derivative
                    dh = 1e-7 * max(1.0, abs(xn))
                    d = (f(xn + dh) - fxn) / dh
                    if abs(d) < 1e-15:
                        break
                    step = fxn / d
                    # Damping if step is too large
                    if abs(step) > 10 * max(1.0, abs(xn)):
                        step = np.sign(step) * 10 * max(1.0, abs(xn))
                    xn_new = xn - step
                    if not np.isfinite(xn_new):
                        break
                    xn = xn_new
                if abs(f(xn)) <= 1e-8:
                    return xn

    except Exception:
        pass

    # ---- Phase 2: Newton's method from x0 with adaptive step control ----
    x = float(x0)
    best_x = x
    best_fx = abs(f(x))
    if best_fx <= 1e-8:
        return x

    for it in range(80):
        if _deadline_exceeded():
            break
        fx = f(x)
        afx = abs(fx)
        if afx < best_fx:
            best_x = x
            best_fx = afx
        if afx <= 1e-8:
            return x

        # Numerical derivative (central difference for accuracy)
        dh = 1e-6 * max(1.0, abs(x))
        fp = f(x + dh)
        fm = f(x - dh)
        d = (fp - fm) / (2.0 * dh)

        if abs(d) < 1e-15:
            # Derivative too small; try a small perturbation
            x = x + 1e-4 * max(1.0, abs(x))
            continue

        step = fx / d
        # Limit step size
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = np.sign(step) * max_step

        x_new = x - step
        if not np.isfinite(x_new):
            break

        # Check for oscillation: if we're going back and forth, damp
        x = x_new

    # ---- Phase 3: Bisection fallback if Newton failed ----
    # Try to find a sign change around x0
    if not _deadline_exceeded():
        # Expand interval until sign change or max size
        a, b = x0 - 1.0, x0 + 1.0
        fa, fb = f(a), f(b)

        # Expand if no sign change
        for _ in range(20):
            if _deadline_exceeded():
                break
            if fa * fb <= 0:
                break
            a = a - (b - a)
            b = b + (b - a)
            fa = f(a)
            fb = f(b)
            if abs(b - a) > 1e6:
                break

        if fa * fb <= 0:
            # Bisection
            for _ in range(60):
                if _deadline_exceeded():
                    break
                mid = 0.5 * (a + b)
                fm = f(mid)
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm <= 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                if abs(b - a) < 1e-12:
                    break

    # Return the best candidate found
    if abs(f(best_x)) <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
