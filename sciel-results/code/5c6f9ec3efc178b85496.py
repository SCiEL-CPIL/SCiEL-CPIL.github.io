<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 145.0
    
    # Try to construct rank-12 decomposition
    # Strategy: partition B (3x4) into two 3x2 blocks
    # For each 2x3x2 sub-problem, try to use 6 terms
    # Total: 12 terms
    
    # Actually, let's first verify the rank-16 solution works,
    # then try to find rank 12 via structural search
    
    def _build_rank16():
=======
import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 145.0
    
    def _build_rank16():
>>>>>>> REPLACE

