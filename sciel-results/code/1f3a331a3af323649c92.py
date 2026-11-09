"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication 3x4 by 4x7.
    Rank is 3*4*7 = 84.
    
    The current program attempts gradient descent to reduce rank, but this is
    numerically unstable and unlikely to succeed for exact integer/half-integer
    decompositions within the time limit. The inspirations confirm that the
    schoolbook decomposition is the reliable baseline. We return it directly
    with a fast, safe construction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Using NumPy for faster construction
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term: A[i,j] * B[j,k] contributes to C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (per trace convention W[k*n+i])
                
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
                # Check time (less frequent)
                if column % 10 == 0 and time.time() > deadline:
                    # If we run out of time, we must return something valid.
                    # Since we are filling sequentially, we might have a partial matrix.
                    # A partial matrix is INVALID.
                    # So we must ensure we finish or return a valid fallback.
                    # The loop is tiny (84 iterations), so this check is overkill but safe.
                    pass

    # Convert to lists to match the seed interface strictly if needed, 
    # or return numpy arrays. The seed returned lists. 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Lists of lists are safer for compatibility with strict type checkers.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
