"""QUADRATIC POLYNOMIAL WITH BIOMEDICAL INTERACTIONS FOR E. COLI GROWTH RATE."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic polynomial basis with biologically-motivated interactions.
    
    10 parameters for:
    - 1 intercept
    - 4 linear terms (b, s, temp, pH)
    - 4 quadratic terms (b², s², temp², pH²)
    - 1 key interaction: s * temp (Monod kinetics: substrate × temperature)
    
    This is linear in params → convex landscape for BFGS → reliable global optimum.
    The s*temp interaction captures the core Monod growth kinetics of E. coli.
    """
    # Normalize inputs to [0, 1] for numerical stability
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0
    
    # Clip for out-of-range stability
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    t_n = np.clip(t_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # Build 10 basis features
    # [1, b, s, t, p, b², s², t², p², s*t]
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
        s_n * t_n  # Monod: substrate concentration × temperature
    ])
    
    # Linear combination with learned weights
    pred = X @ params
    
    return pred

# EVOLVE-BLOCK-END
