import numpy as np
from scipy.special import roots_genlaguerre
import time

_start = time.time()

def construct():
    # m=12, n=1: use roots of generalized Laguerre polynomial L_12^{(1)}(x)
    roots = roots_genlaguerre(12, 1)
    root_locs = sorted(roots.real.tolist())
    return root_locs
