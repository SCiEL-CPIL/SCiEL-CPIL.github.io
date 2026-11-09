"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    """Return the schoolbook decomposition for 3x5x5 matrix multiplication.
    
    The tensor T_{i,j,k} = delta_{j,k} is decomposed as:
    T = sum_{t=1}^{rank} u_t (x) v_t (x) w_t
    where u_t is a vector in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}.
    
    For the schoolbook algorithm, rank = n*m*p and each term corresponds
    to one (i,j,k) triple with j==k.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (as per task contract W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition (optional, for debugging)
    # Check that sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == delta_{j,k}
    # This is guaranteed by construction, so we skip verification for speed.
    
    return result
# EVOLVE-BLOCK-END
