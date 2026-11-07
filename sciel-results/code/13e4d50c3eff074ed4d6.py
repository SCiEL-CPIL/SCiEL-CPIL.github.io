"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Scalar root finder for smooth polynomials.
    Strategy:
      1. Quick Newton phase with adaptive step damping.
      2. If Newton stalls or diverges, use a global bracketing search
         followed by bisection/Newton hybrid.
      3. Final high-precision polish.
    Deadline guard: return best found if wall-clock exceeds 110 s.
    """
    t0 = time.monotonic()
    DEADLINE = 110.0  # seconds, safety margin before 120 s limit

    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    call_count = 1

    def check_deadline():
        return (time.monotonic() - t0) > DEADLINE

    def try_newton(x, fx, max_iter=40):
        """Newton with damping; returns (x, fx, success)."""
        nonlocal call_count
        for _ in range(max_iter):
            if check_deadline():
                return x, fx, False
            if abs(fx) <= 1e-12:
                return x, fx, True
            h = 1e-7 * max(1.0, abs(x))
            fx2 = f(x + h)
            call_count += 1
            d = (fx2 - fx) / h
            if d == 0 or not _is_finite(d):
                return x, fx, False
            step = fx / d
            # Damping: if step is too large, halve it
            if abs(step) > 10.0 * max(1.0, abs(x)):
                step = 10.0 * max(1.0, abs(x)) * (1.0 if step > 0 else -1.0)
            new_x = x - step
            if not _is_finite(new_x):
                return x, fx, False
            new_fx = f(new_x)
            call_count += 1
            # Accept if it improves or is close
            if abs(new_fx) < abs(fx) * 1.5 or abs(new_fx) <= 1e-12:
                x, fx = new_x, new_fx
            else:
                # Try a smaller step
                step *= 0.5
                new_x = x - step
                new_fx = f(new_x)
                call_count += 1
                if abs(new_fx) < abs(fx):
                    x, fx = new_x, new_fx
                else:
                    return x, fx, False
        return x, fx, abs(fx) <= 1e-8

    def _is_finite(v):
        return v == v and v != float('inf') and v != float('-inf')

    # Phase 1: Newton from x0
    x, fx, success = try_newton(x, best_fx)
    if abs(fx) < best_abs_fx:
        best_x, best_fx, best_abs_fx = x, fx, abs(fx)
    if success and abs(fx) <= 1e-8:
        return x

    # Phase 2: Global bracketing search
    # Try expanding brackets around x0
    scale = max(1.0, abs(x0))
    found_bracket = False
    a = b = None
    fa = fb = None

    # First, check if we already have a sign change near current best
    # Try a range of scales
    for s in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if check_deadline():
            break
        lo = x0 - s * scale
        hi = x0 + s * scale
        f_lo = f(lo)
        f_hi = f(hi)
        call_count += 2
        if _is_finite(f_lo) and _is_finite(f_hi):
            if f_lo * f_hi <= 0:
                a, b = lo, hi
                fa, fb = f_lo, f_hi
                found_bracket = True
                break
            # Also check if either endpoint is very close
            if abs(f_lo) < best_abs_fx:
                best_x, best_fx, best_abs_fx = lo, f_lo, abs(f_lo)
            if abs(f_hi) < best_abs_fx:
                best_x, best_fx, best_abs_fx = hi, f_hi, abs(f_hi)
        # Also try asymmetric brackets
        for factor in [0.5, 2.0]:
            if check_deadline():
                break
            lo2 = x0 - s * scale * factor
            hi2 = x0 + s * scale / factor
            f_lo2 = f(lo2)
            f_hi2 = f(hi2)
            call_count += 2
            if _is_finite(f_lo2) and _is_finite(f_hi2):
                if f_lo2 * f_hi2 <= 0:
                    a, b = lo2, hi2
                    fa, fb = f_lo2, f_hi2
                    found_bracket = True
                    break
                if abs(f_lo2) < best_abs_fx:
                    best_x, best_fx, best_abs_fx = lo2, f_lo2, abs(f_lo2)
                if abs(f_hi2) < best_abs_fx:
                    best_x, best_fx, best_abs_fx = hi2, f_hi2, abs(f_hi2)
        if found_bracket:
            break

    if found_bracket and _is_finite(a) and _is_finite(b):
        # Phase 3: Bisection + Newton hybrid
        for _ in range(80):
            if check_deadline():
                break
            mid = (a + b) / 2.0
            f_mid = f(mid)
            call_count += 1
            if not _is_finite(f_mid):
                break
            if abs(f_mid) < best_abs_fx:
                best_x, best_fx, best_abs_fx = mid, f_mid, abs(f_mid)
            if abs(f_mid) <= 1e-12:
                return mid
            if fa * f_mid <= 0:
                b = mid
                fb = f_mid
            else:
                a = mid
                fa = f_mid
            # Try Newton step within bracket
            h = 1e-7 * max(1.0, abs(mid))
            f_mid2 = f(mid + h)
            call_count += 1
            d = (f_mid2 - f_mid) / h
            if _is_finite(d) and d != 0:
                newton_x = mid - f_mid / d
                if a < newton_x < b:
                    f_newton = f(newton_x)
                    call_count += 1
                    if _is_finite(f_newton):
                        if abs(f_newton) < best_abs_fx:
                            best_x, best_fx, best_abs_fx = newton_x, f_newton, abs(f_newton)
                        if abs(f_newton) <= 1e-12:
                            return newton_x
                        if fa * f_newton <= 0:
                            b = newton_x
                            fb = f_newton
                        else:
                            a = newton_x
                            fa = f_newton
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break

    # Phase 4: Final polish from best_x
    if best_abs_fx <= 1e-8:
        # Do a few more Newton steps for precision
        x, fx, _ = try_newton(best_x, best_fx, max_iter=10)
        if abs(fx) <= 1e-8:
            return x
        if abs(fx) < best_abs_fx:
            best_x, best_fx, best_abs_fx = x, fx, abs(fx)

    return best_x
# EVOLVE-BLOCK-END
