import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def _build_rank_32(n, m, p):
    # ... (keep existing)
