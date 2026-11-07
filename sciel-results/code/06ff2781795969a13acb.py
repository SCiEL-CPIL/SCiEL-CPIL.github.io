"""Symbolic regression with 1 input(s) - Hybrid Learnable Sigmoid and Gaussian Basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Hybrid basis with learnable local features.
    
    Parameters (10 total):
      0: Constant offset
      1: Linear coefficient
      2: Sigmoid 1 Amplitude
      3: Sigmoid 1 Slope
      4: Sigmoid 1 Center
      5: Sigmoid 2 Amplitude
      6: Sigmoid 2 Slope
      7: Sigmoid 2 Center
      8: Gaussian 1 Amplitude
      9: Gaussian 1 Center
      
    Basis Functions:
      b0 = 1
      b1 = x
      b2 = A1 * sigmoid(s1 * (x - c1))
      b3 = A2 * sigmoid(s2 * (x - c2))
      b4 = A3 * exp(-((x - c3) / 0.5)^2)
      
    This structure allows the model to fit:
      - Global trends (const, linear)
      - Monotonic steps or S-curves (sigmoids)
      - Local bumps or dips (gaussian)
    The centers and slopes are learnable, providing local flexibility
    that fixed polynomial or tanh bases lack.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Clip x to prevent numerical issues in exp
    x_c = np.clip(x, -100, 100)
    
    p = params
    
    # 1. Constant and Linear
    y = p[0] * np.ones_like(x_c) + p[1] * x_c
    
    # 2. Sigmoid 1
    # sigmoid(z) = 1 / (1 + exp(-z))
    # Clip argument to avoid overflow/underflow in exp
    z1 = p[3] * (x_c - p[4])
    z1 = np.clip(z1, -500, 500)
    y += p[2] / (1.0 + np.exp(-z1))
    
    # 3. Sigmoid 2
    z2 = p[6] * (x_c - p[7])
    z2 = np.clip(z2, -500, 500)
    y += p[5] / (1.0 + np.exp(-z2))
    
    # 4. Gaussian 1
    # Fixed width sigma = 0.5
    diff = (x_c - p[9]) / 0.5
    # Clip squared difference to avoid exp(-large) -> 0 (underflow is ok, but overflow in exp(-neg) is bad if diff is huge? No, diff^2 is positive, so exp(-diff^2) is <= 1)
    # However, if p[9] is huge, x - p[9] is huge, diff is huge, diff^2 is huge, exp(-huge) is 0. Safe.
    y += p[8] * np.exp(-diff**2)
    
    return y
# EVOLVE-BLOCK-END
