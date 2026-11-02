"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
    Uses schoolbook as baseline, then tries to reduce rank via heuristics.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for linear algebra
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor is T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to find a lower-rank decomposition
    
    # Try to use SVD on the "unfolding" of the tensor
    # Unfold along the first mode: T_{i,(j,k)}
    T_unfold = np.zeros((n, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_unfold[i, j * p + k] = 1.0
    
    # SVD
    U_svd, S, Vt = np.linalg.svd(T_unfold, full_matrices=False)
    
    # Check if we can truncate
    # But this gives approximate, not exact
    # For exact, we need integer/half-integer entries
    
    # Given the complexity, return schoolbook
    if time.time() > deadline - 10.0:
        return {"U": U, "V": V, "W": W}
    
    # Try a simple heuristic: remove duplicate terms
    # In schoolbook, all terms are distinct, so no duplicates
    
    # Return schoolbook as fallback
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
