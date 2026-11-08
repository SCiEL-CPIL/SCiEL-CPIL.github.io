"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook():
    """Guaranteed rank-27 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_bilinear(decomp, n, m, p, tol=1e-9, num_trials=10):
    """
    Verify the decomposition by testing with random matrices.
    The decomposition must satisfy:
      C[i,k] = sum_r (sum_{a,b} U[a*m+b, r] * A[a,b]) * 
                           (sum_{b,c} V[b*p+c, r] * B[b,c]) * 
                           W[k*n+i, r]
    equals A @ B for all A, B.
    """
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    np.random.seed(12345)
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        # L_r = U[:, r] dot A.flatten()
        # M_r = V[:, r] dot B.flatten()
        # C[i,k] = sum_r L_r * M_r * W[k*n+i, r]
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        L = U.T @ A_flat  # shape (rank,)
        M = V.T @ B_flat  # shape (rank,)
        
        # C[i,k] = sum_r L[r] * M[r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] = np.dot(L * M, W[k * n + i, :])
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    
    return True


def _try_rank23():
    """
    Attempt to construct a rank-23 decomposition using block decomposition.
    Partition 3x3 into 2x2 and 1x1 blocks.
    This is a heuristic attempt; returns None if verification fails.
    """
    n, m, p = DIMENSIONS
    # Use a known rank-23 construction based on 2x2 Strassen + extensions
    # This is a simplified version; full construction is complex.
    # For now, return None to indicate we don't have a verified rank-23.
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill
    
    # Start with guaranteed schoolbook
    result = _schoolbook()
    
    # Try to improve with rank-23 if available
    rank23 = _try_rank23()
    if rank23 is not None:
        if _verify_bilinear(rank23, *DIMENSIONS):
            result = rank23
        elif time.time() < deadline:
            # Keep schoolbook
            pass
    
    return result

# EVOLVE-BLOCK-END
