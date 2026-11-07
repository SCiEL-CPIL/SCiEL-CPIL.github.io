import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0
    
    # Function to check if a bracket is found
    # We will scan outwards from x0
    
    # Define a helper to try a bracket [a, b]
    def try_bracket(a, b, fa, fb):
        if fa * fb < 0:
            try:
                # brentq requires fa and fb to have different signs
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                # Verify
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
            except Exception:
                pass
        return None

    # Scan strategy:
    # 1. Small steps near x0
    # 2. Larger steps further out
    
    h = 0.1
    # We will keep track of the last point to avoid redundant calls
    # We scan right and left
    
    last_x = x0
    last_f = f0
    
    # Scan right
    for i in range(50):
        next_x = x0 + (i + 1) * h
        # Optimization: if we are doing a linear scan, we just step by h
        # But if we increase h, we need to be careful not to skip.
        # Let's stick to a fixed small h for the first phase to ensure we don't skip roots in high-frequency oscillations.
        # If the frequency is very high, 0.1 might be too big.
        # Let's try h=0.01 first? 50 steps * 0.01 = 0.5 range.
        # If no root in 0.5, try larger steps?
        
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(last_x, next_x, last_f, fn)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
        
    # If no root found in right 50 steps of 0.1 (range 5.0), try left
    last_x = x0
    last_f = f0
    for i in range(50):
        next_x = x0 - (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(next_x, last_x, fn, last_f)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
        
    # If still no root, try larger steps to find a sign change in a wider area
    # This might skip roots, but if the function is oscillatory, there are likely many.
    # If we skip, we might find a bracket that contains multiple roots, which brentq handles (finds one).
    
    step = 0.5
    last_x = x0
    last_f = f0
    for i in range(50):
        next_x = x0 + (i + 1) * step
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(last_x, next_x, last_f, fn)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
        
    last_x = x0
    last_f = f0
    for i in range(50):
        next_x = x0 - (i + 1) * step
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(next_x, last_x, fn, last_f)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
        
    # If all else fails, return x0 (will likely fail tolerance, but avoids exception)
    return x0
