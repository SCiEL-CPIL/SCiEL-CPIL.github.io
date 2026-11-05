import numpy as np
from scipy.special import roots_laguerre
import time

_start_time = time.time()
_DEADLINE = 620.0

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses the zeros of the Laguerre polynomial L_12(x), which are the natural
    root locations for the Cohn-Goncalves Fourier sign uncertainty construction.
    """
    roots = roots_laguerre(12)
    return [float(r) for r in roots]
