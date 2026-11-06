"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using adaptive bracketing + bisection.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    TOL = 1e-8
    MAX_EVALS = 500

    eval_count = 0

    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except:
        best_fx = float('inf')

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    update_best(best_x, best_fx)

    if abs(best_fx) <= TOL:
        return best_x

    # We'll maintain two points (a, fa) and (b, fb) to try to bracket.
    # Start with a = x0, and try to find b such that fa * fb < 0.
    a = best_x
    fa = best_fx

    # Phase 1: Find a bracket by expanding outward from best_x
    b = None
    fb = None
    found_bracket = False

    # Try expanding in both directions with geometric steps
    step_size = max(1.0, abs(best_x)) * 0.1
    max_expansions = 50

    for i in range(max_expansions):
        if not check_time() or eval_count >= MAX_EVALS:
            break

        # Try positive direction
        b_pos = a + step_size
        try:
            fb_pos = call_f(b_pos)
        except:
            fb_pos = float('inf')
        if math.isfinite(fb_pos):
            update_best(b_pos, fb_pos)
            if fa * fb_pos < 0:
                b = b_pos
                fb = fb_pos
                found_bracket = True
                break

        # Try negative direction
        b_neg = a - step_size
        try:
            fb_neg = call_f(b_neg)
        except:
            fb_neg = float('inf')
        if math.isfinite(fb_neg):
            update_best(b_neg, fb_neg)
            if fa * fb_neg < 0:
                b = b_neg
                fb = fb_neg
                found_bracket = True
                break

        # Expand step size
        step_size *= 2.0

        # If we've gone too far, reset step and try from best_x again
        if step_size > 1e6 * max(1.0, abs(best_x)):
            step_size = max(1.0, abs(best_x)) * 0.01
            a = best_x
            fa = best_fx

    if not found_bracket:
        # No bracket found; try secant method as fallback
        if eval_count < MAX_EVALS and check_time():
            x0_sec = float(x0)
            x1_sec = best_x
            fx0 = call_f(x0_sec) if x0_sec != best_x else best_fx
            fx1 = best_fx

            # Ensure we have two distinct points
            if x0_sec == x1_sec:
                x0_sec = x1_sec + 1e-6
                try:
                    fx0 = call_f(x0_sec)
                except:
                    fx0 = float('inf')

            for i in range(30):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                if abs(fx1) <= TOL:
                    return x1_sec

                if fx0 == fx1 or not math.isfinite(fx0) or not math.isfinite(fx1):
                    break

                x_new = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
                try:
                    fx_new = call_f(x_new)
                except:
                    break

                if not math.isfinite(fx_new):
                    break

                update_best(x_new, fx_new)

                # Check if we now have a bracket
                if fx0 * fx_new < 0:
                    a, fa = x0_sec, fx0
                    b, fb = x_new, fx_new
                    found_bracket = True
                    break
                elif fx1 * fx_new < 0:
                    a, fa = x1_sec, fx1
                    b, fb = x_new, fx_new
                    found_bracket = True
                    break

                x0_sec, fx0 = x1_sec, fx1
                x1_sec, fx1 = x_new, fx_new

            if abs(fx1) <= TOL:
                return x1_sec
            if abs(best_fx) <= TOL:
                return best_x

    # Phase 2: If we have a bracket, use bisection (guaranteed convergence)
    if found_bracket and eval_count < MAX_EVALS and check_time():
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Bisection
        for i in range(100):
            if not check_time() or eval_count >= MAX_EVALS:
                break

            mid = (a + b) / 2.0
            try:
                fmid = call_f(mid)
            except:
                break

            if not math.isfinite(fmid):
                break

            update_best(mid, fmid)

            if abs(fmid) <= TOL:
                return mid

            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid

        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b

    # Return best point found
    return best_x
# EVOLVE-BLOCK-END
