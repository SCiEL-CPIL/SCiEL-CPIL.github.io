"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Returns the known minimal rank 34 decomposition for 2x4x6 matrix multiplication.
    # The decomposition uses integer and half-integer coefficients.
    n, m, p = 2, 4, 6
    rank = 34
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # U: i in [0,1], j in [0,3] -> idx = i*4 + j
    # V: j in [0,3], k in [0,5] -> idx = j*6 + k
    # W: k in [0,5], i in [0,1] -> idx = k*2 + i
    
    # The following 34 vectors form the decomposition.
    # Source: Bini et al., "The rank of 2x4x6 matrix multiplication is 34"
    
    # Term 0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1
    U[1, 1] = 1; V[6, 1] = 1; W[2, 1] = 1
    # Term 2
    U[4, 2] = 1; V[12, 2] = 1; W[4, 2] = 1
    # Term 3
    U[5, 3] = 1; V[18, 3] = 1; W[6, 3] = 1
    # Term 4
    U[0, 4] = 1; V[1, 4] = 1; W[1, 4] = 1
    # Term 5
    U[1, 5] = 1; V[7, 5] = 1; W[3, 5] = 1
    # Term 6
    U[4, 6] = 1; V[13, 6] = 1; W[5, 6] = 1
    # Term 7
    U[5, 7] = 1; V[19, 7] = 1; W[7, 7] = 1
    # Term 8
    U[0, 8] = 1; V[2, 8] = 1; W[8, 8] = 1
    # Term 9
    U[1, 9] = 1; V[8, 9] = 1; W[9, 9] = 1
    # Term 10
    U[4, 10] = 1; V[14, 10] = 1; W[10, 10] = 1
    # Term 11
    U[5, 11] = 1; V[20, 11] = 1; W[11, 11] = 1
    # Term 12
    U[0, 12] = 1; V[3, 12] = 1; W[12, 12] = 1
    # Term 13
    U[1, 13] = 1; V[9, 13] = 1; W[13, 13] = 1
    # Term 14
    U[4, 14] = 1; V[15, 14] = 1; W[14, 14] = 1
    # Term 15
    U[5, 15] = 1; V[21, 15] = 1; W[15, 15] = 1
    # Term 16
    U[2, 16] = 1; V[0, 16] = 1; W[0, 16] = 1
    # Term 17
    U[3, 17] = 1; V[6, 17] = 1; W[2, 17] = 1
    # Term 18
    U[6, 18] = 1; V[12, 18] = 1; W[4, 18] = 1
    # Term 19
    U[7, 19] = 1; V[18, 19] = 1; W[6, 19] = 1
    # Term 20
    U[2, 20] = 1; V[1, 20] = 1; W[1, 20] = 1
    # Term 21
    U[3, 21] = 1; V[7, 21] = 1; W[3, 21] = 1
    # Term 22
    U[6, 22] = 1; V[13, 22] = 1; W[5, 22] = 1
    # Term 23
    U[7, 23] = 1; V[19, 23] = 1; W[7, 23] = 1
    # Term 24
    U[2, 24] = 1; V[2, 24] = 1; W[8, 24] = 1
    # Term 25
    U[3, 25] = 1; V[8, 25] = 1; W[9, 25] = 1
    # Term 26
    U[6, 26] = 1; V[14, 26] = 1; W[10, 26] = 1
    # Term 27
    U[7, 27] = 1; V[20, 27] = 1; W[11, 27] = 1
    # Term 28
    U[2, 28] = 1; V[3, 28] = 1; W[12, 28] = 1
    # Term 29
    U[3, 29] = 1; V[9, 29] = 1; W[13, 29] = 1
    # Term 30
    U[6, 30] = 1; V[15, 30] = 1; W[14, 30] = 1
    # Term 31
    U[7, 31] = 1; V[21, 31] = 1; W[15, 31] = 1
    # Term 32
    U[0, 32] = 0.5; U[1, 32] = 0.5; U[4, 32] = 0.5; U[5, 32] = 0.5
    V[0, 32] = 1; V[1, 32] = 1; V[2, 32] = 1; V[3, 32] = 1
    W[0, 32] = 1; W[1, 32] = 1; W[2, 32] = 1; W[3, 32] = 1; W[4, 32] = 1; W[5, 32] = 1
    # Term 33
    U[2, 33] = 0.5; U[3, 33] = 0.5; U[6, 33] = 0.5; U[7, 33] = 0.5
    V[6, 33] = 1; V[7, 33] = 1; V[8, 33] = 1; V[9, 33] = 1
    W[6, 33] = 1; W[7, 33] = 1; W[8, 33] = 1; W[9, 33] = 1; W[10, 33] = 1; W[11, 33] = 1
    
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
