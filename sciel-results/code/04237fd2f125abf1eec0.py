import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 2 Gaussians (with bounded centers and stable widths) + Cubic Polynomial.
    Params:
    0: a1 (Gaussian 1 amplitude)
    1: c1_raw (raw center for Gaussian 1, squashed to [-2, 2])
    2: log_w1 (log of Gaussian 1 width)
    3: a2 (Gaussian 2 amplitude)
    4: c2_raw (raw center for Gaussian 2, squashed to [-2, 2])
    5: log_w2 (log of Gaussian 2 width)
    6: b0 (Poly constant)
    7: b1 (Poly linear)
    8: b2 (Poly quadratic)
    9: b3 (Poly cubic)
    """
    # Unpack parameters
    a1 = params[0]
    c1_raw = params[1]
    log_w1 = params[2]
    a2 = params[3]
    c2_raw = params[4]
    log_w2 = params[5]
    b0 = params[6]
    b1 = params[7]
    b2 = params[8]
    b3 = params[9]
    
    # Squash centers to [-2, 2] to keep them within the domain of interest
    # tanh maps real numbers to (-1, 1). Scaling by 2 maps to (-2, 2).
    # This helps BFGS by keeping centers in a relevant range and providing smooth gradients.
    c1 = 2.0 * np.tanh(c1_raw)
    c2 = 2.0 * np.tanh(c2_raw)
    
    # Widths
    # Clip log_w to prevent extreme widths which can cause numerical instability
    # or make the optimization landscape very flat/sharp.
    # exp(5) ~ 148 (very wide), exp(-5) ~ 0.0067 (very narrow)
    log_w1 = np.clip(log_w1, -5, 5)
    log_w2 = np.clip(log_w2, -5, 5)
    w1 = np.exp(log_w1)
    w2 = np.exp(log_w2)
    
    # Calculate Gaussians
    # Use stable computation: exp(-0.5 * ((x-c)/w)^2)
    # Clip the argument to exp to avoid overflow (though negative args are safe for exp,
    # large positive args would overflow, but here arg is always <= 0).
    # However, if w is extremely small, (x-c)/w can be huge, leading to huge negative arg.
    # exp(-huge) -> 0. This is fine.
    # But to be safe against any numerical quirks, we clip the argument.
    g1_arg = -0.5 * ((x - c1) / w1) ** 2
    g2_arg = -0.5 * ((x - c2) / w2) ** 2
    
    # Clip to [-50, 50]. Since arg is always negative, this effectively clips to [-50, 0].
    # exp(-50) is ~1.9e-22, which is negligible.
    g1_arg = np.clip(g1_arg, -50, 50)
    g2_arg = np.clip(g2_arg, -50, 50)
    
    g1 = a1 * np.exp(g1_arg)
    g2 = a2 * np.exp(g2_arg)
    
    # Calculate Polynomial
    poly = b0 + b1 * x + b2 * x**2 + b3 * x**3
    
    # Sum
    y = g1 + g2 + poly
    
    return y
