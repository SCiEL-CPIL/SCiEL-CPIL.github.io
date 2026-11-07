"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Function Model (Padé Approximant style).
    
    This model uses a ratio of two 4th-degree polynomials:
    y(x) = P(x) / Q(x)
    
    where P(x) = sum(a_i * x^i) and Q(x) = sum(b_i * x^i).
    
    params[0..4] are coefficients for the numerator P.
    params[5..9] are coefficients for the denominator Q.
    
    To handle the scale invariance of the rational function (P/Q = kP/kQ),
    we normalize the denominator coefficients to unit norm during computation.
    This improves the conditioning for BFGS.
    
    We use a soft denominator to avoid division by zero.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Split parameters
    a = params[0:5] # Numerator coefficients
    b = params[5:10] # Denominator coefficients
    
    # Normalize denominator coefficients to prevent scale invariance issues
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        # If b is zero, set to a small valid state to avoid division by zero
        # This case is unlikely during optimization if initialized to 1s
        b_norm = 1.0
        b_safe = np.ones(5)
    else:
        b_safe = b / b_norm
        
    # Compute powers of x
    # x is shape (n,)
    x0 = np.ones_like(x)
    x1 = x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Compute Numerator P(x)
    # P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    P = a[0]*x0 + a[1]*x1 + a[2]*x2 + a[3]*x3 + a[4]*x4
    
    # Compute Denominator Q(x)
    # Q(x) = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
    Q = b_safe[0]*x0 + b_safe[1]*x1 + b_safe[2]*x2 + b_safe[3]*x3 + b_safe[4]*x4
    
    # Soft denominator to avoid division by zero
    # Use sign-preserving square root of Q^2 + eps^2
    eps = 1e-8
    Q_soft = np.sign(Q) * np.sqrt(Q*Q + eps*eps)
    # Handle case where Q is exactly 0 (sign is 0)
    Q_soft = np.where(Q == 0, eps, Q_soft)
    
    # Compute result
    result = P / Q_soft
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass # Return current result
        
    return result
# EVOLVE-BLOCK-END
