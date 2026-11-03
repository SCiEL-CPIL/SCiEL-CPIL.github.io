"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook rank
    r_school = n * m * p
    # Build schoolbook matrices
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    
    # Use vectorized construction for speed, though loop is fast enough
    # This ensures we are well within time limits
    indices = list(product(range(n), range(m), range(p)))
    for c, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0

    def _pack(U, V, W, r):
        return U[:, :r].tolist(), V[:, :r].tolist(), W[:, :r].tolist()

    # Check time before attempting any complex search (none planned, but good practice)
    if time.time() - start_time > TIME_LIMIT:
        return {"U": _pack(U_sb, V_sb, W_sb, r_school)[0], 
                "V": _pack(U_sb, V_sb, W_sb, r_school)[1], 
                "W": _pack(U_sb, V_sb, W_sb, r_school)[2]}

    # Return the schoolbook decomposition as the safe, verified baseline.
    U_list, V_list, W_list = _pack(U_sb, V_sb, W_sb, r_school)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
