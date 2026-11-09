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
    1. Evaluate f(x0).
    2. Use Secant Method (1 call/iteration) to converge to the root.
    3. Fallback to Bisection if Secant diverges or fails.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    # Initial check
    fx0 = _safe_f(f, x0)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0

    if not math.isfinite(fx0):
        # If x0 is invalid, try 0
        x0 = 0.0
        fx0 = _safe_f(f, x0)
        if not math.isfinite(fx0):
            return x0 # Best effort

    if abs(fx0) <= 1e-8:
        return x0

    # Phase 1: Secant Method
    # We need two initial points. We use x0 and a slightly perturbed x0.
    # The perturbation size is chosen to be small enough for local linearity
    # but large enough to avoid division by zero or numerical noise.
    # For polynomials, a relative perturbation is usually good.
    x1 = x0 + 1.0
    fx1 = _safe_f(f, x1)
    
    # If the perturbation was bad (e.g., x0 was huge and x1 overflowed or didn't change sign),
    # try a smaller relative step or a different direction.
    if not math.isfinite(fx1) or abs(x1 - x0) < 1e-12:
        x1 = x0 - 1.0
        fx1 = _safe_f(f, x1)

    if not math.isfinite(fx1) or abs(x1 - x0) < 1e-12:
        # Try a relative step
        x1 = x0 * 1.01 if x0 != 0 else 1.0
        fx1 = _safe_f(f, x1)

    if not math.isfinite(fx1) or abs(x1 - x0) < 1e-12:
        # Last resort: just use bisection with a wide bracket?
        # But for now, let's try secant with whatever we have.
        pass

    # Secant iterations
    # Superlinear convergence order ~1.618
    # 20 iterations is usually enough for double precision
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break

        # Check if we've converged
        if abs(fx1) <= 1e-8:
            return x1

        # If fx1 is not finite, we are in trouble. Try to recover?
        if not math.isfinite(fx1):
            break

        # Calculate next point
        denom = fx1 - fx0
        if abs(denom) < 1e-30:
            # Derivative is effectively zero or points are too close.
            # Perturb x1 slightly and restart secant logic?
            # Or just break and fallback.
            break

        dx = -fx1 * (x1 - x0) / denom
        x2 = x1 + dx
        fx2 = _safe_f(f, x2)

        # Update
        x0, fx0 = x1, fx1
        x1, fx1 = x2, fx2

        # If x2 is not finite, break
        if not math.isfinite(x1):
            break

        # If we have a bracket from previous steps, we could switch to bisection,
        # but pure secant is faster if it stays in the basin of attraction.
        # For polynomials, if it diverges, it usually goes to infinity quickly.

    # If Secant succeeded
    if math.isfinite(fx1) and abs(fx1) <= 1e-8:
        return x1

    # Phase 2: Fallback to Bisection
    # We need a bracket. We might have one from the secant attempts,
    # but we didn't track it explicitly in the loop above.
    # Let's do a quick bracketing search around the last good point or x0.
    
    # Let's restart bracketing from the original x0 to be safe, or use the last finite point.
    # Using the last finite point is better if it's close.
    a = x0 if math.isfinite(x0) else 0.0
    fa = _safe_f(f, a)
    
    if not math.isfinite(fa):
        a = 0.0
        fa = _safe_f(f, a)
        if not math.isfinite(fa):
            return a

    if abs(fa) <= 1e-8:
        return a

    b = None
    fb = None
    scale = 1.0
    found_bracket = False

    # Expand window geometrically
    while time.time() < deadline - 2.0 and scale < 1e12:
        # Try positive direction
        b_pos = a + scale
        fb_pos = _safe_f(f, b_pos)
        if math.isfinite(fb_pos) and fa * fb_pos < 0:
            b = b_pos
            fb = fb_pos
            found_bracket = True
            break
        
        # Try negative direction
        b_neg = a - scale
        fb_neg = _safe_f(f, b_neg)
        if math.isfinite(fb_neg) and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            found_bracket = True
            break
            
        scale *= 2.0

    if not found_bracket:
        # If no bracket found, return best guess
        return a

    # Bisection
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    for _ in range(60): # Log2(1e12) ~ 40, 60 is safe
        if time.time() > deadline - 1.0:
            break
        m = (a + b) / 2.0
        fm = _safe_f(f, m)
        
        if not math.isfinite(fm):
            break
            
        if abs(fm) <= 1e-8:
            return m
            
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
            
        if abs(b - a) < 1e-12:
            break
            
    return (a + b) / 2.0

# EVOLVE-BLOCK-END
