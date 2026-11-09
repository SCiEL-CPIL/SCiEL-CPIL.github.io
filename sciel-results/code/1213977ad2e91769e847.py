"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS
    # Rank 1 decomposition for the all-ones tensor
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
