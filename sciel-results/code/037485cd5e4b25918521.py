"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    # Expected improvement with a small exploration parameter (xi=0.01)
    # to encourage more exploration and avoid premature convergence.
    xi = 0.01
    s = np.maximum(sigma, 1e-8)
    z = (f_best - mu - xi) / s
    ei = (f_best - mu - xi) * 0.5 * (1.0 + np.vectorize(__import__('math').erf)(z / np.sqrt(2.0))) + \
         s * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
    return np.maximum(ei, 0.0)
# EVOLVE-BLOCK-END
