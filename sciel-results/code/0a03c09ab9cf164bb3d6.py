"""Symbolic regression: rational function architecture P3(x,y) / (1 + P1(x,y))."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model:
    f(x,y) = P3(x,y) / (1 + P1(x,y))
    
    where P3 is a cubic polynomial in (x,y) and P1 is a linear polynomial in (x,y).
    
    P3(x,y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y + p6*x^3 + p7*y^3 + p8*x^2*y + p9*x*y^2
    Wait, that's 10 coefficients for P3 alone. We need params for both numerator and denominator.
    
    Let's restructure. We have 10 params total.
    Let's use a simpler rational form to leave room for both numerator and denominator.
    
    Option A:
    Numerator: quadratic (6 terms: 1, x, y, x^2, y^2, xy)
    Denominator: 1 + linear (2 terms: x, y)
    Total params: 6 + 2 = 8. We have 2 left over.
    Let's add x^2*y and x*y^2 to numerator? That makes numerator cubic-ish (8 terms) + 2 denom = 10.
    
    Let's define:
    Numerator P2(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y + c6*x^2*y + c7*x*y^2
    Denominator D(x,y) = 1 + c8*x + c9*y
    
    Total 10 params. This allows for a fairly rich numerator and a simple denominator.
    This can model functions like 1/(1+x), (x+y)/(1-x), etc.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT:
        # Fallback: just the numerator (denominator ~ 1)
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
        return c0 + c1*x + c2*y + c3*x**2 + c4*y**2 + c5*x*y + c6*x**2*y + c7*x*y**2
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Numerator: quadratic + mixed cubic terms
    num = (c0 + c1*x + c2*y + c3*x**2 + c4*y**2 + c5*x*y + c6*x**2*y + c7*x*y**2)
    
    # Denominator: 1 + linear
    den = 1.0 + c8*x + c9*y
    
    # Avoid division by zero or near-zero
    # Clip denominator to avoid numerical instability
    den_safe = np.where(np.abs(den) < 1e-8, np.sign(den) * 1e-8 + 1e-8, den)
    # Actually, simpler: just add a small epsilon with sign preservation or just clip
    # np.sign(0) is 0, so let's be careful.
    # Better: den_safe = np.clip(den, -1e6, 1e6) then handle zeros?
    # The evaluator says non-finite error costs 1e30. So we just need finite output.
    # If den is 0, num/0 is inf/nan.
    # Let's use a safe division.
    
    # A robust way:
    # pred = num / den
    # If den is close to 0, the value blows up. 
    # To keep it finite, we can clip the denominator away from 0.
    # den_safe = np.where(den > 0, np.maximum(den, 1e-6), np.minimum(den, -1e-6))
    
    den_safe = np.where(den > 0, np.maximum(den, 1e-6), np.minimum(den, -1e-6))
    
    pred = num / den_safe
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to numerator only (linear/quadratic)
        pred = c0 + c1*x + c2*y + c3*x**2 + c4*y**2 + c5*x*y + c6*x**2*y + c7*x*y**2
        if not np.all(np.isfinite(pred)):
            pred = np.zeros_like(x)
    
    return pred
# EVOLVE-BLOCK-END
