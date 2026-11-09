"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import newton

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
    1. Expanding window search to find a bracket [a, b] where f(a) and f(b) have opposite signs.
    2. Use scipy.optimize.newton with finite difference derivative for fast quadratic convergence.
    3. Fallback to bisection if Newton fails.
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

    # Phase 1: Find a bracket [a, b] with a sign change
    a = x0
    fa = fx0
    if not math.isfinite(fa):
        # If x0 is invalid, try 0
        a = 0.0
        fa = _safe_f(f, a)
        if not math.isfinite(fa):
            return a # Best effort

    if abs(fa) <= 1e-8:
        return a

    b = None
    fb = None
    scale = 1.0
    found_bracket = False
    
    # Expand window geometrically
    while time.time() < deadline - 5.0 and scale < 1e12:
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
        # If no bracket found, we can't use Newton safely or Bisection.
        # Return best guess (x0) or a point with small |f| if we tracked it?
        # For now, return x0 as fallback.
        return a

    # Phase 2: Refine using Newton's Method
    # scipy.optimize.newton uses finite difference derivative by default if fprime is None
    # We limit maxiter to ensure we don't waste calls if it diverges or stalls
    try:
        # tol=1e-12 is tighter than requirement 1e-8 to be safe
        # maxiter=20 is usually enough for quadratic convergence from a good bracket
        root = newton(f, a, fprime=None, tol=1e-12, maxiter=20, full_output=False)
        
        # Verify the root
        f_root = _safe_f(f, root)
        if math.isfinite(f_root) and abs(f_root) <= 1e-8:
            return root
    except Exception:
        pass

    # Phase 3: Fallback to Bisection if Newton failed
    # We have a valid bracket [a, b] with fa * fb < 0
    # Ensure a < b for bisection logic
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    for _ in range(60): # Log2(1e12) ~ 40, 60 is safe
        if time.time() > deadline - 2.0:
            break
        m = (a + b) / 2.0
        fm = _safe_f(f, m)
        
        if not math.isfinite(fm):
            # If midpoint is invalid, shrink bracket?
            # For polynomials, this shouldn't happen inside a valid bracket unless we hit overflow.
            # Just break and return midpoint
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
