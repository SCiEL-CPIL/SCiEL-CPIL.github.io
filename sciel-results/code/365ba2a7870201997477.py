<<<<<<< SEARCH
import numpy as np
import time
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
=======
import numpy as np
import time
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [2, 4, 7]

def _mat_to_floats(mat):
    """Convert a matrix of Fractions/ints to a matrix of floats/ints for JSON serialization."""
    return [[float(x) if isinstance(x, Fraction) else x for x in row] for row in mat]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
>>>>>>> REPLACE

