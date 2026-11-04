import numpy as np
from scipy.special import roots_laguerre

def construct():
        # The task specifies m=12. We use the roots of the Laguerre polynomial of order 12.
        # These roots are the optimal nodes for the Laguerre-Gaussian quadrature and
        # are expected to minimize the uncertainty constant in the Cohn-Goncalves framework.
        roots, _ = roots_laguerre(12)
        # Return as a list of floats
        return [float(r) for r in roots]
