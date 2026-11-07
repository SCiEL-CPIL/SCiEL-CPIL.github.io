import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator.
    Uses a scaled polynomial basis to improve conditioning for BFGS.
    Inputs:
        x: array of positions
        v: array of velocities
        params: array of 10 fitted constants
    Output:
        array of accelerations
    """
    # Scale inputs to be roughly in [-1, 1] or similar range to help optimizer
    # x range ~ [-0.57, 0.53] -> *2 gives ~ [-1.1, 1.0]
    # v range ~ [-0.17, 0.28] -> *10 gives ~ [-1.7, 2.8]
    xs = x * 2.0
    vs = v * 10.0
    
    # Pre-compute powers
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs5 = xs3 * xs2
    
    vs2 = vs * vs
    vs3 = vs2 * vs
    vs5 = vs3 * vs2
    
    # Basis functions:
    # 0: 1
    # 1: xs
    # 2: vs
    # 3: xs^3
    # 4: vs^3
    # 5: xs^2 * vs
    # 6: xs * vs^2
    # 7: xs^5
    # 8: vs^5
    # 9: xs^2 * vs^3
    
    pred = np.zeros_like(x, dtype=float)
    
    pred += params[0] * 1.0
    pred += params[1] * xs
    pred += params[2] * vs
    pred += params[3] * xs3
    pred += params[4] * vs3
    pred += params[5] * (xs2 * vs)
    pred += params[6] * (xs * vs2)
    pred += params[7] * xs5
    pred += params[8] * vs5
    pred += params[9] * (xs2 * vs3)
    
    return pred
