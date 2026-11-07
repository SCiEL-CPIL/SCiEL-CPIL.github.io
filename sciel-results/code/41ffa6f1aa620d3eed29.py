"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a sum of Gaussian radial basis functions (RBFs)
    to recover an unknown function. This approach uses local basis functions
    which can fit complex, non-polynomial shapes effectively.
    
    Parameters are mapped to ensure stability:
    - Amplitudes are direct parameters.
    - Centers are mapped via tanh to stay within [-2, 2].
    - Widths are mapped via exp to ensure positivity and avoid zero width.
    """
    # Unpack parameters
    # 3 Gaussians + 1 offset = 10 parameters
    # G1: p0 (amp), p1 (center_raw), p2 (width_raw)
    # G2: p3 (amp), p4 (center_raw), p5 (width_raw)
    # G3: p6 (amp), p7 (center_raw), p8 (width_raw)
    # Offset: p9
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate centers using tanh to keep them bounded in [-2, 2]
    c1 = 2.0 * np.tanh(p1)
    c2 = 2.0 * np.tanh(p4)
    c3 = 2.0 * np.tanh(p7)
    
    # Calculate widths using exp to ensure positivity
    # Clip raw width params to prevent overflow in exp
    w1 = 0.1 + 0.1 * np.exp(np.clip(p2, -10, 10))
    w2 = 0.1 + 0.1 * np.exp(np.clip(p5, -10, 10))
    w3 = 0.1 + 0.1 * np.exp(np.clip(p8, -10, 10))
    
    # Calculate Gaussian terms
    # Using a safe exponent calculation
    g1 = p0 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    g2 = p3 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    g3 = p6 * np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    # Sum with offset
    return g1 + g2 + g3 + p9
# EVOLVE-BLOCK-END
