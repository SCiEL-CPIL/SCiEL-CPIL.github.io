"""SEARCH: generate many candidate equations, fit each, keep the best."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Flexible Quadratic Polynomial Basis.
    
    Uses 10 parameters for a quadratic polynomial with:
    - 1 intercept
    - 4 linear terms (b, s, temp, pH)
    - 4 quadratic terms (b², s², temp², pH²)
    - 1 key interaction term (b * temp)
    
    This creates a convex optimization landscape for BFGS (linear in params),
    eliminating local minima while capturing curvature and main interactions.
    """
    # Normalize inputs to [0, 1] range for numerical stability
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0
    
    # Build polynomial feature matrix
    # Features: [1, b, s, t, p, b², s², t², p², b*t]
    ones = np.ones_like(b_n)
    
    X = np.column_stack([
        ones,
        b_n,
        s_n,
        t_n,
        p_n,
        b_n ** 2,
        s_n ** 2,
        t_n ** 2,
        p_n ** 2,
        b_n * t_n  # Key interaction: nutrient * temperature
    ])
    
    # Linear combination: prediction = X @ params
    mu = X @ params
    
    return mu

# EVOLVE-BLOCK-END
