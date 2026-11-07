"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 150s wall-clock.
_START_TIME = time.time()
_MAX_TIME = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis function combining Chebyshev polynomials (normalized) and
    diverse non-polynomial functions (trig, hyperbolic, exponential).
    
    The input x is in [-3, 3]. We normalize a subset to [-1, 1] for Chebyshev
    stability, but use raw x for other basis functions where scaling is natural
    or less critical (like sin/cos which are periodic, or tanh which is bounded).
    
    Basis:
    0: 1 (constant)
    1: z (normalized x)
    2: T_2(z) (Chebyshev 2nd order)
    3: T_3(z) (Chebyshev 3rd order)
    4: T_4(z) (Chebyshev 4th order)
    5: sin(x)
    6: cos(x)
    7: exp(-x^2)
    8: tanh(x)
    9: x * sin(x)
    
    This provides a rich manifold for BFGS to fit various hidden functions.
    """
    # Check time limit
    if time.time() - _START_TIME > _MAX_TIME:
        # Return a safe default if we are about to time out
        # The last valid state is not easily accessible in a stateless function,
        # but returning zeros is a safe fallback that won't crash.
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Normalize x for Chebyshev basis
    z = x / 3.0
    
    # Pre-allocate basis matrix or compute components
    # We compute components individually for clarity and to avoid large temporary arrays
    # if we were to use a matrix approach, but 10 terms is small.
    
    b0 = np.ones_like(x)
    
    # Chebyshev terms on z
    # T_0 = 1, T_1 = z, T_2 = 2z^2 - 1, T_3 = 4z^3 - 3z, T_4 = 8z^4 - 8z^2 + 1
    # Let's just compute the recurrence up to T_4 for the first 5 slots.
    # Actually, the current program uses T_0..T_9. The inspirations use a mix.
    # Let's stick to a mix that covers both low-order polynomial-like (Cheb) and specific features.
    
    # Chebyshev T_1(z)
    b1 = z
    
    # Chebyshev T_2(z) = 2*z^2 - 1
    z2 = z * z
    b2 = 2.0 * z2 - 1.0
    
    # Chebyshev T_3(z) = 4*z^3 - 3*z
    z3 = z2 * z
    b3 = 4.0 * z3 - 3.0 * z
    
    # Chebyshev T_4(z) = 8*z^4 - 8*z^2 + 1
    z4 = z2 * z2
    b4 = 8.0 * z4 - 8.0 * z2 + 1.0
    
    # Non-polynomial features on raw x
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_neg_x2 = np.exp(-x * x)
    tanh_x = np.tanh(x)
    x_sin_x = x * sin_x
    
    b5 = sin_x
    b6 = cos_x
    b7 = exp_neg_x2
    b8 = tanh_x
    b9 = x_sin_x
    
    # Linear combination
    # params is length 10
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
