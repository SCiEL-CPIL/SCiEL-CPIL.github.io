# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using an additive neural network structure.
    
    f(x) = p0 + p1*x + p2*tanh(p3*x + p4) + p5*tanh(p6*x + p7) + p8*tanh(p9*x + p10)
    
    Parameters:
    p0: Bias
    p1: Linear coefficient
    p2, p5, p8: Amplitudes for the three tanh units
    p3, p6, p9: Slopes for the three tanh units
    p4, p7, p10: Biases (centers) for the three tanh units
    """
    p = np.asarray(params, dtype=float)
    x = np.asarray(x, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.full_like(x, np.nan)
        
    # Linear part
    y = p[0] + p[1] * x
    
    # Tanh units
    # Clip arguments to prevent potential issues, though tanh is stable
    arg1 = np.clip(p[3] * x + p[4], -500, 500)
    arg2 = np.clip(p[6] * x + p[7], -500, 500)
    arg3 = np.clip(p[9] * x + p[10], -500, 500)
    
    y += p[2] * np.tanh(arg1)
    y += p[5] * np.tanh(arg2)
    y += p[8] * np.tanh(arg3)
    
    y = np.where(np.isfinite(y), y, 0.0)
    return y
# EVOLVE-BLOCK-END
