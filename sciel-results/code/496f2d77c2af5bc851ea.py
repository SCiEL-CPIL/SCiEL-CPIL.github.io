import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Returns the schoolbook decomposition (rank 125).
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    
    def build_schoolbook():
        """Build the standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        
        return rank, U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """
        Verify that U, V, W form a valid matrix multiplication decomposition.
        """
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        max_err = 0.0
        for a_row in range(n):
            for a_col in range(m):
                for b_row in range(m):
                    for b_col in range(p):
                        expected = 1.0 if a_col == b_row else 0.0
                        u_val = U[a_row * m + a_col, :]
                        v_val = V[b_row * p + b_col, :]
                        w_val = W[b_col * n + a_row, :]
                        val = np.dot(u_val * v_val, w_val)
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
                        if max_err > tol:
                            return False, max_err
        
        return True, max_err
    
    # Build schoolbook decomposition
    rank, U, V, W = build_schoolbook()
    
    # Verify it's correct
    valid, err = verify_decomposition(U, V, W)
    
    # Check deadline before returning
    if time.time() > deadline - safety_margin:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
