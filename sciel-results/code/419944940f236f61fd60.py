"""Root finder for 3 fixed oscillatory equations using bracketing + Illinois method."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    MAX_CALLS = 480     # Stay well under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        val = f(x)
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = x
        return val

    def illinois(fa, a, fb, b):
        """Illinois method: bisection + linear interpolation."""
        nonlocal calls
        for _ in range(200):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            # Illinois: linear interpolation, but if it's outside [a,b], use midpoint
            if fa == fb:
                mid = (a + b) / 2.0
            else:
                mid = (a * fb - b * fa) / (fb - fa)
            if not (a < mid < b):
                mid = (a + b) / 2.0
            fm = _f(mid)
            if math.isnan(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                # Illinois trick: shrink the side with smaller |f|
                if abs(fa) < abs(fb):
                    fa = fa / 2.0
                else:
                    fb = fb / 2.0
                a = mid
                fa = fm
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # Phase 1: Find a bracket around x0
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Try expanding in both directions to find a sign change
    step = 0.1 * max(1.0, abs(x))
    found_bracket = False

    # Try positive direction
    for i in range(1, 50):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_pos = x + i * step
        f_pos = _f(x_pos)
        if math.isnan(f_pos):
            break
        if abs(f_pos) <= 1e-8:
            return x_pos
        if fx * f_pos < 0:
            # Found bracket [x, x_pos]
            result = illinois(fx, x, f_pos, x_pos)
            return result
        step *= 1.5  # Expand adaptively

    # Try negative direction
    step = 0.1 * max(1.0, abs(x))
    for i in range(1, 50):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_neg = x - i * step
        f_neg = _f(x_neg)
        if math.isnan(f_neg):
            break
        if abs(f_neg) <= 1e-8:
            return x_neg
        if fx * f_neg < 0:
            # Found bracket [x_neg, x]
            result = illinois(f_neg, x_neg, fx, x)
            return result
        step *= 1.5

    # Phase 2: If no bracket found, do a finer local scan
    # Try a dense local grid around x0
    local_step = 0.01 * max(1.0, abs(x))
    prev_x = x - 10 * local_step
    prev_fx = _f(prev_x)
    if not math.isnan(prev_fx) and abs(prev_fx) <= 1e-8:
        return prev_x

    for i in range(21):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        cur_x = x - 10 * local_step + i * local_step
        cur_fx = _f(cur_x)
        if math.isnan(cur_fx):
            break
        if abs(cur_fx) <= 1e-8:
            return cur_x
        if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
            result = illinois(prev_fx, prev_x, cur_fx, cur_x)
            return result
        prev_x = cur_x
        prev_fx = cur_fx

    # Return best x found
    return best_x

# EVOLVE-BLOCK-END
