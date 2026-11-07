"""Symbolic regression using stable Chebyshev rational function approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation using Chebyshev polynomials:
    y = N(t) / D(t)
    
    where t = x/2 scales input to [-1, 1].
    
    Numerator N(t) is a degree-5 Chebyshev polynomial:
    N(t) = sum_{i=0}^{5} p_i * T_i(t)
    
    Denominator D(t) is a bounded perturbation of 1:
    D(t) = 1.0 + 0.2 * sum_{j=1}^{4} tanh(q_j) * T_j(t)
    
    The tanh and scaling (0.2) ensure D(t) remains strictly positive 
    (approx range [0.2, 1.8]) even if params are large, preventing 
    singularities and ensuring BFGS stability.
    
    params layout (length 10):
      params[0..5] = numerator coefficients p0..p5
      params[6..9] = denominator coefficients q1..q4 (mapped via tanh)
    """
    start_time = time.time()
    DEADLINE = 145.0
    
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Scale x to [-1, 1]
    t = x * 0.5
    
    # Compute Chebyshev polynomials T_0 to T_5
    T_0 = np.ones_like(t)
    T_1 = t
    
    # Recurrence: T_{n+1} = 2*t*T_n - T_{n-1}
    T_2 = 2.0 * t * T_1 - T_0
    T_3 = 2.0 * t * T_2 - T_1
    T_4 = 2.0 * t * T_3 - T_2
    T_5 = 2.0 * t * T_4 - T_3
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5 = params[0], params[1], params[2], params[3], params[4], params[5]
    q1, q2, q3, q4 = params[6], params[7], params[8], params[9]
    
    # Compute Numerator
    numerator = (p0 * T_0 + p1 * T_1 + p2 * T_2 + 
                 p3 * T_3 + p4 * T_4 + p5 * T_5)
    
    # Compute Denominator
    # Use tanh to bound the influence of q_j, ensuring D(t) stays close to 1
    # and strictly positive.
    # Max deviation: 0.2 * (1*1 + 1*1 + 1*1 + 1*1) = 0.8.
    # So D(t) in [0.2, 1.8].
    denom_perturbation = (0.2 * (np.tanh(q1) * T_1 + 
                                 np.tanh(q2) * T_2 + 
                                 np.tanh(q3) * T_3 + 
                                 np.tanh(q4) * T_4))
    
    denominator = 1.0 + denom_perturbation
    
    # Compute prediction
    result = numerator / denominator
    
    # Check time
    if time.time() > DEADLINE:
        pass
        
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
# EVOLVE-BLOCK-END
