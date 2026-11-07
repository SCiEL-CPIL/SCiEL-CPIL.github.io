"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Mixed basis: polynomials (cubic), trig (2 freq), and 2 parametric Gaussians.
    # All terms are O(1)-scaled so BFGS from all-ones stays well-conditioned.
    # 10 params: [a0..a3 poly, b1,b2 trig, c1,s1,w1,c2,s2,w2]

    # --- Polynomial part (degree 0..3) ---
    p0 = np.ones_like(x)
    p1 = x
    p2 = x * x
    p3 = p2 * x

    # --- Trig part (frequencies scaled to ~1 rad over [-3,3]) ---
    s1 = np.sin(x)
    s2 = np.sin(2.0 * x)

    # --- Gaussian bumps: exp(-w * (x - s)^2) ---
    # w and s are params; clamp width to stay positive & finite.
    w1 = np.maximum(params[7], 1e-3)
    w2 = np.maximum(params[9], 1e-3)
    g1 = np.exp(-w1 * (x - params[6]) ** 2)
    g2 = np.exp(-w2 * (x - params[8]) ** 2)

    return (params[0] * p0 +
            params[1] * p1 +
            params[2] * p2 +
            params[3] * p3 +
            params[4] * s1 +
            params[5] * s2 +
            params[6] * g1 +
            params[7] * g2 +
            params[8] * np.cos(x) +
            params[9] * np.cos(2.0 * x))
# EVOLVE-BLOCK-END
