import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket
    # Start with small step
    h = 1e-4
    # Max expansion steps to avoid infinite loop
    max_steps = 20 
    
    a = None
    b = None
    fa = None
    fb = None
    
    # We need to find a, b such that f(a)*f(b) < 0
    # We check x-h, x, x+h
    
    # First, check immediate neighbors
    # But we already have fx.
    
    # Strategy: Expand outwards from x0
    # Keep track of last point to reuse
    last_x = x
    last_f = fx
    
    for i in range(max_steps):
        if time.time() > deadline:
             # Return best guess? Or raise? 
             # The task says "an exception... makes that equation a failure".
             # Returning a bad x makes it a failure.
             # Better to return the x with smallest |f(x)| seen so far?
             # But if |f(x)| > 1e-8, it fails anyway.
             # So just return x0 or last_x.
             return last_x
             
        # Try positive side
        x_pos = last_x + h
        f_pos = f(x_pos)
        if abs(f_pos) <= 1e-8:
            return x_pos
        if last_f * f_pos < 0:
            a = last_x
            b = x_pos
            fa = last_f
            fb = f_pos
            break
        # Try negative side
        x_neg = last_x - h
        f_neg = f(x_neg)
        if abs(f_neg) <= 1e-8:
            return x_neg
        if last_f * f_neg < 0:
            a = x_neg
            b = last_x
            fa = f_neg
            fb = last_f
            break
        
        # Update last point? 
        # If we just expand h, we are checking further and further.
        # But we might skip a sign change if we jump over it?
        # No, if f(last_x) and f(last_x + h) have same sign, and f(last_x) and f(last_x - h) have same sign,
        # we haven't found a bracket around last_x.
        # We should expand the search range.
        # It is more efficient to keep x0 fixed and expand h?
        # Or move the "last_x" to the edge?
        # If we move last_x to x_pos, we lose the info about the interval [last_x, x_pos] having no sign change?
        # Actually, if f(last_x) and f(x_pos) have same sign, there might still be a root between them (2 roots).
        # But for oscillatory, usually 1 root per period.
        # Safer: Keep x0 as anchor?
        # If we keep x0 as anchor, we check [x0-h, x0+h].
        # If no sign change, increase h.
        # This ensures we don't miss a root close to x0.
        
        h *= 2.0
        
    if a is None:
        # No bracket found. Try a wider search or fallback
        # Maybe the root is very far.
        # Or maybe the function doesn't change sign (tangent root).
        # For oscillatory, sign change is likely.
        # If we fail to bracket, we might try minimization of |f(x)|?
        # But that's expensive.
        # Let's try one more thing: check if we can use Newton on the last point?
        # Or just return x0 (failure).
        return x0

    # Use brentq
    try:
        root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        return root
    except Exception:
        # Fallback to bisection manually if brentq fails?
        # Or return midpoint?
        return (a + b) / 2
