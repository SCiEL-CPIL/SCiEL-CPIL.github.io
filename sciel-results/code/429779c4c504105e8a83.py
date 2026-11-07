"""Derivative-free safeguarded inverse-interpolation root finder."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    # --- Step 1: Efficient Bracketing ---
    a = x0
    fa = f(a)
    if abs(fa) <= 1e-8:
        return a
    if not math.isfinite(fa):
        a = 0.0
        fa = f(a)
        if abs(fa) <= 1e-8:
            return a

    b = x0 + 1.0
    fb = f(b)
    if abs(fb) <= 1e-8:
        return b
    if not math.isfinite(fb):
        b = x0 - 1.0
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b

    # If no sign change with [x0, x0+1], try [x0-1, x0]
    if math.isfinite(fa) and math.isfinite(fb) and fa * fb > 0.0:
        c = x0 - 1.0
        fc = f(c)
        if abs(fc) <= 1e-8:
            return c
        if math.isfinite(fc):
            if fa * fc < 0.0:
                a, b = c, a
                fa, fb = fc, fa
            elif fb * fc < 0.0:
                a, b = c, b
                fa, fb = fc, fb
            else:
                # Still no bracket, keep [a, b] as is for expansion loop
                # But we have a new point c. 
                # It's better to expand from the wider interval [c, b] if it contains the root?
                # We don't know where the root is.
                # Let's just let the expansion loop handle it, but we have wasted 1 call.
                # To mitigate, we can set a, b to c, b if |fc| < |fa|?
                # No, just proceed.
                pass

    # Expand bracket if no sign change
    # Limit to 40 expansions to save calls, relying on the fact that
    # roots of these polynomials are likely within a reasonable range.
    for _ in range(40):
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
            break
        # Expand the side with the smaller absolute function value?
        # Or simply expand the interval size.
        # Let's expand the side that is "smaller" in magnitude to catch nearby roots?
        # Actually, standard expansion is to widen the interval.
        if abs(fa) <= abs(fb):
            # fa is smaller, maybe root is closer to a? Expand a further?
            # Or expand b to cover more ground?
            # Let's just expand the interval symmetrically or exponentially.
            # Current logic: expand the side farther from x0?
            # Let's try: if |fa| < |fb|, expand a, else expand b.
            new_a = a - (b - a)
            fa = f(new_a)
            if abs(fa) <= 1e-8: return new_a
            if not math.isfinite(fa):
                a = x0
                fa = f(a)
            else:
                a = new_a
        else:
            new_b = b + (b - a)
            fb = f(new_b)
            if abs(fb) <= 1e-8: return new_b
            if not math.isfinite(fb):
                b = x0 + 1.0
                fb = f(b)
            else:
                b = new_b
        
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
            break
    else:
        # If no bracket found, return the point with smallest |f|
        # This is a failure case for the contract but we must return a float
        return a if abs(fa) <= abs(fb) else b

    # --- Step 2: Safeguarded Regula Falsi (Brent-like) ---
    # We have a bracket [a, b] with fa*fb < 0
    # To ensure convergence, we maintain that a is the "better" point (smaller |f|)
    # This helps with the safeguard condition.
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    for _ in range(50):
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        # If the bracket is very small, return the midpoint
        if abs(b - a) < 1e-10:
            return 0.5 * (a + b)

        # Propose new point using Inverse Linear Interpolation
        # x_new = b - fb * (b - a) / (fb - fa)
        if fa == fb:
            # Should not happen if bracketed, but guard against it
            x_new = 0.5 * (a + b)
        else:
            x_new = b - fb * (b - a) / (fb - fa)

        # Safeguard:
        # 1. Must be inside (a, b)
        # 2. Must not be too close to the "good" endpoint (a) to avoid stalling.
        #    Condition: |x_new - a| > 0.5 * |b - a|
        #    This ensures we make significant progress.
        if not (a < x_new < b) or (abs(x_new - a) <= 0.5 * abs(b - a)):
            x_new = 0.5 * (a + b)

        f_new = f(x_new)
        if abs(f_new) <= 1e-8:
            return x_new
        
        if not math.isfinite(f_new):
            # If evaluation fails, fall back to bisection step
            # This shouldn't happen for polynomials, but safe guard
            x_new = 0.5 * (a + b)
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new

        # Update bracket
        if fa * f_new < 0.0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new

        # Re-order to keep 'a' as the better point (smaller |f|)
        # This is crucial for the safeguard condition to work correctly
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Return the best estimate
    return a if abs(fa) <= abs(fb) else b
# EVOLVE-BLOCK-END
