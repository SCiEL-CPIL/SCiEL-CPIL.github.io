"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Compute zeros of the Laguerre polynomial L_12(x)
    # These are the natural double-root locations for the Cohn-Goncalves construction
    x, w = np.polynomial.laguerre.laggauss(12)
    return [float(xi) for xi in x]
# EVOLVE-BLOCK-END
