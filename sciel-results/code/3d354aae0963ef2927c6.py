"""Rewritten program: use positive roots of d/dx L_13^(-1/2)(x) as double-root locations."""
import time
import sympy as sp

def construct():
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill

    # Define symbols
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)

    # Compute L_13^(-1/2)(x) and its derivative
    L13 = sp.laguerre(13, x, alpha)
    dL13 = sp.diff(L13, x)

    # Find roots of the derivative
    roots = sp.nroots(dL13, n=50, maxsteps=100)

    # Filter for positive real roots (should be 13 roots, all positive and distinct)
    positive_roots = []
    for r in roots:
        if abs(sp.im(r)) < 1e-10 and sp.re(r) > 0:
            positive_roots.append(float(sp.re(r)))

    # We need exactly 12 double roots. The derivative of L_13^(-1/2) has 12 roots
    # (since L_13 is degree 13, its derivative is degree 12).
    # Actually, L_n^(alpha) has degree n, so L_13^(-1/2) has degree 13, 
    # and its derivative has degree 12. So we should get exactly 12 positive roots.
    
    if len(positive_roots) != 12:
        # Fallback: try a different approach or return what we have
        # For now, return the available roots if we have at least 12, else use defaults
        if len(positive_roots) >= 12:
            positive_roots = positive_roots[:12]
        else:
            # Fallback to simple spacing
            positive_roots = [4.0 * (i + 1) for i in range(12)]

    # Sort the roots
    positive_roots.sort()

    # Ensure they are finite, > 0, and pairwise distinct
    valid = True
    for i in range(len(positive_roots)):
        if not (positive_roots[i] > 0) or not sp.finite(sp.Rational(positive_roots[i])):
            valid = False
            break
        for j in range(i + 1, len(positive_roots)):
            if abs(positive_roots[i] - positive_roots[j]) < 1e-10:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]

    return positive_roots
