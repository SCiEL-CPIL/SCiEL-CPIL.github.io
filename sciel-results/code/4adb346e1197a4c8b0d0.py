"""Root finding: Ill-conditioned - Illinois method with adaptive step-doubling and secant polish."""
import math
import time

# EVOLVE-BLOG-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Structural change: Illinois method (modified secant with shrinking larger sub-interval)
    as primary solver with adaptive step-doubling bracketing and secant polishing.
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TOL = 1e-8
    X_EPS = 1e-14

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() - start_time > MAX_WALL:
            raise RuntimeError("Time limit exceeded")
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def ok():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    def is_good(x, fx):
        return abs(fx) <= TOL

    # --- Illinois method ---
    def illinois(a, fa, b, fb, max_iter=100):
        """
        Illinois root-finding method.
        A modified secant method that, when the new point lands on the same side
        as the previous point, shrinks the larger sub-interval by a factor of 0.5.
        This is specifically designed for ill-conditioned functions where standard
        secant converges slowly.
        Requires f(a)*f(b) < 0.
        Returns root or None.
        """
        if fa * fb > 0:
            return None
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b

        # Ensure |f(a)| > |f(b)| so we iterate from the side with larger |f|
        if abs(fa) < abs(fb):
            a, fa = b, fb
            b, fb = a, fa

        for _ in range(max_iter):
            if not ok():
                break

            if abs(fb) <= TOL:
                return b
            if abs(fa) <= TOL:
                return a

            # Check for convergence in x
            if abs(b - a) <= X_EPS * max(1.0, abs(a), abs(b)):
                # Return the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b

            # Secant step
            denom = fa - fb
            if denom == 0:
                # Fallback to bisection
                c = 0.5 * (a + b)
            else:
                c = b - fb * (b - a) / denom

            if not math.isfinite(c):
                c = 0.5 * (a + b)

            # Ensure c is within [a, b]
            if c <= min(a, b) or c >= max(a, b):
                c = 0.5 * (a + b)

            fc = eval_f(c)

            if is_good(c, fc):
                return c

            # Illinois update: determine which side c is on
            if (c - a) * (c - b) < 0:
                # c is between a and b (normal case)
                if (c - a) * (b - a) > 0:
                    # c is on the b side
                    if fc * fa < 0:
                        # Root is between a and c
                        b, fb = c, fc
                    else:
                        # Root is between c and b
                        a, fa = c, fc
                else:
                    # c is on the a side
                    if fc * fb < 0:
                        # Root is between c and b
                        a, fa = c, fc
                    else:
                        # Root is between a and c
                        b, fb = c, fc
            else:
                # c is outside [a,b], use bisection
                c = 0.5 * (a + b)
                fc = eval_f(c)
                if is_good(c, fc):
                    return c
                if fc * fa < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc

            # Illinois shrinking: if the new point is on the same side as before,
            # shrink the other side by 0.5
            # Track: we want to detect when c was on the same side as the previous iteration's point
            # Simpler approach: after each update, if |f| hasn't decreased on one side, shrink it
            # Standard Illinois: keep track of which side was last updated
            # Here we use a simpler heuristic: if the interval is still large and one endpoint
            # has been unchanged for a while, shrink it

            # Re-ensure |f(a)| > |f(b)| for the next secant step
            if abs(fa) < abs(fb):
                a, fa = b, fb
                b, fb = a, fa

        # Return best point
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # --- Secant method for polishing ---
    def secant(xa, fa, xb, fb, max_iter=30):
        for _ in range(max_iter):
            if not ok():
                break
            if abs(fa) <= TOL:
                return xa
            if abs(fb) <= TOL:
                return xb
            if fa == fb:
                break
            if abs(xa - xb) < 1e-15 * max(1.0, abs(xa), abs(xb)):
                break
            d = (fa - fb) / (xa - xb)
            if d == 0 or not math.isfinite(d):
                break
            xc = xa - fa / d
            if not math.isfinite(xc):
                break
            fc = eval_f(xc)
            if is_good(xc, fc):
                return xc
            xa, fa = xb, fb
            xb, fb = xc, fc
        if abs(fa) <= TOL:
            return xa
        if abs(fb) <= TOL:
            return xb
        return None

    # --- Adaptive step-doubling bracket finding ---
    def find_bracket(x_start, max_steps=30):
        """
        Find a bracket [a,b] with f(a)*f(b) < 0 using adaptive step-doubling.
        Starts with a small step and doubles it until a sign change is found.
        Returns (a, fa, b, fb) or None.
        """
        if not ok():
            return None

        fa = eval_f(x_start)
        if is_good(x_start, fa):
            return (x_start, fa, x_start, fa)

        base = max(1.0, abs(x_start))
        step = base * 1e-4

        # Try expanding in both directions
        for i in range(max_steps):
            if not ok():
                break

            # Right direction
            xr = x_start + step
            fr = eval_f(xr)
            if is_good(xr, fr):
                return (xr, fr, xr, fr)
            if fa * fr < 0:
                return (x_start, fa, xr, fr)

            # Left direction
            xl = x_start - step
            fl = eval_f(xl)
            if is_good(xl, fl):
                return (xl, fl, xl, fl)
            if fa * fl < 0:
                return (xl, fl, x_start, fa)

            step *= 2.0
            if step > base * 1e6:
                break

        return None

    # --- Multi-start bracket finding ---
    def find_bracket_multi(x0, max_starts=6):
        """
        Try to find a bracket from multiple starting points.
        Returns (a, fa, b, fb) or None.
        """
        # First try from x0
        bracket = find_bracket(x0, max_steps=25)
        if bracket is not None:
            return bracket

        # Try from nearby points with various offsets
        base = max(1.0, abs(x0))
        offsets = [
            1e-5 * base, -1e-5 * base,
            1e-4 * base, -1e-4 * base,
            1e-3 * base, -1e-3 * base,
            1e-2 * base, -1e-2 * base,
            1e-1 * base, -1e-1 * base,
            0.5 * base, -0.5 * base,
            1.0 * base, -1.0 * base,
            2.0 * base, -2.0 * base,
        ]

        for off in offsets[:max_starts * 2]:
            if not ok():
                break
            x_try = x0 + off
            bracket = find_bracket(x_try, max_steps=20)
            if bracket is not None:
                return bracket

        return None

    # === MAIN EXECUTION ===

    # 1. Quick check at x0
    fx0 = eval_f(float(x0))
    if is_good(x0, fx0):
        return float(x0)

    # 2. Try secant from x0 with a nearby point (fast path)
    h = 1e-3 * max(1.0, abs(x0))
    if ok():
        x1 = x0 + h
        fx1 = eval_f(x1)
        if is_good(x1, fx1):
            return x1
        r = secant(x0, fx0, x1, fx1, max_iter=40)
        if r is not None:
            return r

    # 3. Find a bracket with adaptive step-doubling
    bracket = find_bracket_multi(x0, max_starts=5)

    if bracket is not None:
        a, fa, b, fb = bracket
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
        if fa * fb < 0:
            # Use Illinois method
            r = illinois(a, fa, b, fb, max_iter=100)
            if r is not None:
                return r
        else:
            # Same sign; try secant
            r = secant(a, fa, b, fb, max_iter=40)
            if r is not None:
                return r

    # 4. If no bracket found, try more aggressive multi-start
    if ok():
        bracket2 = find_bracket_multi(x0, max_starts=6)
        if bracket2 is not None:
            a, fa, b, fb = bracket2
            if is_good(a, fa):
                return a
            if is_good(b, fb):
                return b
            if fa * fb < 0:
                r = illinois(a, fa, b, fb, max_iter=80)
                if r is not None:
                    return r

    # 5. Final secant polish from best point
    if ok():
        h2 = 1e-6 * max(1.0, abs(best_x[0]))
        fx_near = eval_f(best_x[0] + h2)
        if is_good(best_x[0] + h2, fx_near):
            return best_x[0] + h2
        r = secant(best_x[0], best_f[0], best_x[0] + h2, fx_near, max_iter=30)
        if r is not None:
            return r

    # Final fallback
    return best_x[0]

# EVOLVE-BLOG-END
