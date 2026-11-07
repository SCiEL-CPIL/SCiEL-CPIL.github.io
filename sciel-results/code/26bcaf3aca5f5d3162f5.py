"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a physics-informed basis.
    
    Uses a basis inspired by the Duffing equation structure:
    {x, x^3, v, v|v|, x^2v, x|v|, sin(x), sin(3x), v sin(x), 1}
    This aligns the feature space with expected nonlinear restoring forces
    and velocity-dependent damping, providing a well-conditioned landscape
    for BFGS optimization.
    """
    p = params
    
    # Construct the feature matrix F with shape (N, 10)
    # F[:, i] corresponds to params[i]
    
    # 1. x
    f1 = x
    # 2. x^3
    f2 = x ** 3
    # 3. v
    f3 = v
    # 4. v * |v|
    f4 = v * np.abs(v)
    # 5. x^2 * v
    f5 = (x ** 2) * v
    # 6. x * |v|
    f6 = x * np.abs(v)
    # 7. sin(x)
    f7 = np.sin(x)
    # 8. sin(3x)
    f8 = np.sin(3.0 * x)
    # 9. v * sin(x)
    f9 = v * np.sin(x)
    # 10. 1 (bias)
    f10 = np.ones_like(x)
    
    # Stack features into a matrix (N, 10)
    F = np.column_stack((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10))
    
    # Weighted sum: F @ p
    return F @ p

# EVOLVE-BLOCK-END
