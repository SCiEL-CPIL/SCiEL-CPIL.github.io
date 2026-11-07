import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    
    # Check initial point
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0
    
    # Function to check if a bracket is found
    def try_bracket(a, b, fa, fb):
        if fa * fb < 0:
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
            except Exception:
                pass
        return None

    # Strategy: Use adaptive scanning with smaller steps for high-frequency oscillations
    # Start with very small steps, then expand
    
    # Phase 1: Very fine scan near x0 (handles high frequency oscillations)
    h = 0.01
    last_x = x0
    last_f = f0
    
    # Scan right with fine steps
    for i in range(100):  # 100 steps * 0.01 = 1.0 range
        next_x = x0 + (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(last_x, next_x, last_f, fn)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
    
    # Scan left with fine steps
    last_x = x0
    last_f = f0
    for i in range(100):
        next_x = x0 - (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(next_x, last_x, fn, last_f)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
    
    # Phase 2: Medium steps
    h = 0.05
    last_x = x0
    last_f = f0
    
    # Scan right
    for i in range(100):  # 100 steps * 0.05 = 5.0 range
        next_x = x0 + (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(last_x, next_x, last_f, fn)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
    
    # Scan left
    last_x = x0
    last_f = f0
    for i in range(100):
        next_x = x0 - (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(next_x, last_x, fn, last_f)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
    
    # Phase 3: Larger steps to cover wider range
    h = 0.2
    last_x = x0
    last_f = f0
    
    # Scan right
    for i in range(100):  # 100 steps * 0.2 = 20.0 range
        next_x = x0 + (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(last_x, next_x, last_f, fn)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
    
    # Scan left
    last_x = x0
    last_f = f0
    for i in range(100):
        next_x = x0 - (i + 1) * h
        fn = f(next_x)
        if last_f * fn < 0:
            res = try_bracket(next_x, last_x, fn, last_f)
            if res is not None:
                return res
        last_x = next_x
        last_f = fn
    
    # If all else fails, return x0
    return x0
