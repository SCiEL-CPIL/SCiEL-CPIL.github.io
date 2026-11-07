"""Rewritten program with a rational function model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a rational function y = P(x) / Q(x).
    P and Q are polynomials of degree 4.
    params[0:5] are coefficients of P (constant to x^4).
    params[5:10] are coefficients of Q (constant to x^4).
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Compute powers of x
    x0 = np.ones_like(x)
    x1 = x
    x2 = x**2
    x3 = x**3
    x4 = x**4
    
    # Numerator P(x)
    # p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    numerator = params[0]*x0 + params[1]*x1 + params[2]*x2 + params[3]*x3 + params[4]*x4
    
    # Denominator Q(x)
    # q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4
    denominator = params[5]*x0 + params[6]*x1 + params[7]*x2 + params[8]*x3 + params[9]*x4
    
    # Add a small epsilon to denominator to avoid division by zero
    # This is a heuristic to keep the function finite during optimization
    # If the true function has a pole, this will approximate it with a large value.
    # If not, it acts as a regularizer.
    eps = 1e-8
    # We want to preserve the sign of the denominator if possible, 
    # but simply adding eps might shift it.
    # A safer approach for optimization stability:
    # denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    # But this is non-smooth.
    # Let's just add eps to the square of the denominator? No, that changes the function.
    # Let's just add eps to the denominator directly. 
    # Since initial denominator is positive, this is fine.
    # If denominator becomes negative, adding eps reduces the magnitude.
    # To be robust, we can use:
    # denom_safe = denominator + eps * np.sign(denominator)
    # But sign is non-smooth.
    
    # Alternative: Just do the division and let numpy handle inf/nan.
    # The evaluator says "non-finite error costs 1e30".
    # If BFGS hits a singularity, it will get a high cost and move away.
    # However, if it gets stuck in a region where denominator is 0, it might fail.
    
    # A common trick is to add a small constant to the denominator.
    # Let's try:
    # denom_safe = denominator + 1e-6
    # This shifts the pole.
    
    # Better: use a smooth approximation that is always non-zero?
    # No, rational functions need to be able to go to infinity.
    
    # Let's stick to the direct division but handle the zero case by replacing zeros with a small value.
    # This is technically non-smooth at zero, but zero is a measure-zero set.
    # BFGS uses gradients. If we replace 0 with eps, the gradient is 0 there, which might be okay.
    
    # Actually, the most standard way in these "black box" evaluations is to just divide.
    # If the result is inf, the loss is huge, and the optimizer will likely step back.
    
    # However, to be safe against "failed fit", we should ensure we don't return nan/inf if possible.
    # Let's use a small epsilon addition to the denominator.
    # This effectively regularizes the denominator away from zero.
    
    denominator_safe = denominator + 1e-8
    
    prediction = numerator / denominator_safe
    
    return prediction
# EVOLVE-BLOCK-END
