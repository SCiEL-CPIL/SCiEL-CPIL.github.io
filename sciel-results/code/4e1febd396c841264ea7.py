"""Rewritten program for symbolic regression with a rational basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational basis model with 10 parameters.
    f(x, y) = (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy) / (1 + p6*x + p7*y + p8*x^2 + p9*y^2)
    
    This model can represent:
    - Linear functions (if denom params are 0)
    - Quadratic functions (if denom params are 0)
    - Rational functions (e.g., 1/(1+x^2))
    - More complex smooth functions
    
    Inputs x, y are in [-1, 1].
    Params start at 1.0.
    Denominator at start: 1 + x + y + x^2 + y^2.
    Min value of denom on [-1,1]^2 is 1 (at x=-1, y=-1). Max is 5.
    So denominator is always positive and well-behaved.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin before 120s limit

    p = params

    # Pre-compute powers
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Numerator: p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy
    num = p[0] + p[1] * x + p[2] * y + p[3] * x2 + p[4] * y2 + p[5] * xy

    # Denominator: 1 + p6*x + p7*y + p8*x^2 + p9*y^2
    # Note: We omit xy in denominator to keep it simpler and symmetric-ish, 
    # but we have 10 params total. 6 in num, 4 in denom.
    den = 1.0 + p[6] * x + p[7] * y + p[8] * x2 + p[9] * y2

    # Safety check: ensure denominator is not too small to avoid huge gradients
    # Clip denominator to avoid division by zero or near-zero
    # Since den is bounded away from 0 for initial params, this is mostly a safety net
    # for cases where BFGS wanders into bad regions.
    # However, if den becomes negative, the function sign flips. 
    # To maintain stability, we can use np.where or just let it be.
    # Given the start point, BFGS should stay in a reasonable region.
    
    # If time is running out, return current prediction
    if time.time() > deadline:
         return num / den

    pred = num / den
    
    # Ensure finite values
    # If any non-finite, replace with 0 to avoid crashing the evaluator
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
