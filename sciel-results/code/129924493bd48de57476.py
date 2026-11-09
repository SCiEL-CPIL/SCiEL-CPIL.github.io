"""Symbolic regression: nonlinear oscillator acceleration via physics-informed basis.

Uses a fixed basis of 10 terms derived from standard nonlinear oscillator dynamics:
- Bias
- Linear, quadratic, cubic terms in x (stiffness nonlinearity)
- Linear, quadratic terms in v (damping nonlinearity)
- Cross term x*v
- First and second harmonic forcing terms (sin(t), cos(t), sin(2t))

This basis is linear in the 10 parameters, allowing BFGS to find the optimal
least-squares fit efficiently without complex hyperparameter search.
"""
# EVOLVE-BLOCK-START
import numpy as np

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using physics-informed basis.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    n = len(t)
    
    # 1. Bias
    f0 = np.ones(n)
    
    # 2. x terms: x, x^2, x^3
    f1 = x
    f2 = x * x
    f3 = x * x * x
    
    # 3. v terms: v, v^2
    f4 = v
    f5 = v * v
    
    # 4. Cross term: x*v
    f6 = x * v
    
    # 5. Forcing terms: sin(t), cos(t), sin(2t)
    f7 = np.sin(t)
    f8 = np.cos(t)
    f9 = np.sin(2 * t)
    
    # Stack features
    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out
