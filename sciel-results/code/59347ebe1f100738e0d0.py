"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for conditioning
    z = x / 3.0

    # 10 diverse basis functions:
    # 0: constant
    # 1: linear
    # 2: quadratic
    # 3: cubic
    # 4: quartic
    # 5: sin(pi * z)
    # 6: cos(pi * z)
    # 7: sin(2*pi*z)
    # 8: exp(-z^2)
    # 9: tanh(z)
    p = params
    out = (p[0] +
           p[1] * z +
           p[2] * z**2 +
           p[3] * z**3 +
           p[4] * z**4 +
           p[5] * np.sin(np.pi * z) +
           p[6] * np.cos(np.pi * z) +
           p[7] * np.sin(2.0 * np.pi * z) +
           p[8] * np.exp(-(z * z)) +
           p[9] * np.tanh(z))
    return out
# EVOLVE-BLOCK-END
