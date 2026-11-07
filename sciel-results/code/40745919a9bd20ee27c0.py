"""Chebyshev-activated product basis symbolic regression candidate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x from [-3, 3] to [-1, 1] for stable Chebyshev evaluation
    t = x / 3.0
    
    # Build Chebyshev polynomials T_0 through T_4 using recurrence
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    
    # Build product features: Chebyshev * nonlinear activation
    # 5 features using tanh, 5 features using sin
    # Feature 0: T0 * tanh(x)
    # Feature 1: T1 * tanh(x)
    # Feature 2: T2 * tanh(x)
    # Feature 3: T3 * tanh(x)
    # Feature 4: T4 * tanh(x)
    # Feature 5: T0 * sin(x)
    # Feature 6: T1 * sin(x)
    # Feature 7: T2 * sin(x)
    # Feature 8: T3 * sin(x)
    # Feature 9: T4 * sin(x)
    
    tanh_x = np.tanh(x)
    sin_x = np.sin(x)
    
    features = np.column_stack([
        T0 * tanh_x,
        T1 * tanh_x,
        T2 * tanh_x,
        T3 * tanh_x,
        T4 * tanh_x,
        T0 * sin_x,
        T1 * sin_x,
        T2 * sin_x,
        T3 * sin_x,
        T4 * sin_x
    ])  # (N, 10)
    
    return features @ params
# EVOLVE-BLOCK-END
