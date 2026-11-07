"""Rewritten program for symbolic regression with a rational basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Enhanced rational basis model with 10 parameters.
    f(x, y) = (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy + p6*x^3 + p7*y^3) / (1 + p8*x + p9*y)
    
    This model can represent:
    - Linear functions (if higher-order params are 0)
    - Quadratic and cubic functions
    - Simple rational functions
    
    Inputs x, y are in [-1, 1].
    Params start at 1.0.
    Denominator at start: 1 + x + y.
    Min value of denom on [-1,1]^2 is -1 (at x=-1, y=-1), Max is 3.
    To ensure stability, we add a small constant to keep denominator bounded away from 0.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin before 120s limit

    p = params

    # Pre-compute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y

    # Numerator: p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy + p6*x^3 + p7*y^3
    num = (p[0] + p[1] * x + p[2] * y + p[3] * x2 + p[4] * y2 + 
           p[5] * xy + p[6] * x3 + p[7] * y3)

    # Denominator: 1 + p8*x + p9*y
    # Add small constant to ensure denominator stays away from zero
    den = 1.0 + p[8] * x + p[9] * y
    
    # Safety: ensure denominator doesn't get too close to zero
    # Use a soft approach: if |den| < epsilon, replace with epsilon * sign(den)
    epsilon = 1e-6
    abs_den = np.abs(den)
    sign_den = np.where(den >= 0, 1.0, -1.0)
    safe_den = np.where(abs_den < epsilon, epsilon * sign_den, den)

    # If time is running out, return current prediction
    if time.time() > deadline:
         return num / safe_den

    pred = num / safe_den
    
    # Ensure finite values
    # If any non-finite, replace with 0 to avoid crashing the evaluator
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
