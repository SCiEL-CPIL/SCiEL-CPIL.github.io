"""
Strategy: Hybrid Polynomial-Parametric Basis.
We combine a fixed low-degree polynomial backbone (linear in parameters, stable) 
with two learnable sine-wave terms (non-linear in parameters). 
The 10 parameters are:
[0] p0: Constant offset
[1] p1: Coefficient for x
[2] p2: Coefficient for y
[3] p3: Coefficient for (x^2 + y^2)
[4] A1: Amplitude for Sine 1
[5] k1: Frequency x for Sine 1
[6] l1: Frequency y for Sine 1
[7] A2: Amplitude for Sine 2
[8] k2: Frequency x for Sine 2
[9] l2: Frequency y for Sine 2

This allows the model to adapt its oscillatory structure to the hidden function,
going beyond the capability of a fixed linear basis.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid polynomial and parametric sine basis.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float).flatten()
    
    if params.size != 10:
        return np.zeros_like(x, dtype=float)
    
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    A1, k1, l1 = params[4], params[5], params[6]
    A2, k2, l2 = params[7], params[8], params[9]
    
    # Polynomial backbone
    poly_part = p0 + p1 * x + p2 * y + p3 * (x**2 + y**2)
    
    # Parametric sine terms
    # Clip frequencies to prevent extreme oscillations that might cause numerical issues or slow convergence
    k1 = np.clip(k1, -10, 10)
    l1 = np.clip(l1, -10, 10)
    k2 = np.clip(k2, -10, 10)
    l2 = np.clip(l2, -10, 10)
    
    sine1 = A1 * np.sin(k1 * x + l1 * y)
    sine2 = A2 * np.sin(k2 * x + l2 * y)
    
    pred = poly_part + sine1 + sine2
    
    # Ensure finite output
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
