"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # All other entries are 0.
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Compute tensor product sum
        # T_reconstructed[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can do this efficiently by iterating over c
        T_rec = np.zeros_like(T_target)
        for c in range(U.shape[1]):
            # Outer product of U[:, c], V[:, c], W[:, c]
            # This is a 3D array of shape (n*m, m*p, p*n)
            # We can use broadcasting
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T_target, atol=tol, rtol=0)
    
    # Helper to try to find a decomposition of a given rank using random initialization and gradient-free optimization
    # Since the problem is small, we can use a simple coordinate descent or random perturbation approach.
    # However, exact solutions with integer/half-integer entries are rare for random search.
    # Instead, we will try to find a solution with real entries first, then check if it can be rounded.
    # But the problem asks for integer/half-integer entries.
    # Let's stick to the schoolbook if we can't find a better one quickly.
    
    # For now, we will just return the schoolbook decomposition to ensure validity.
    # A more sophisticated search would be needed to find a lower rank decomposition.
    
    # Schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u, col] = 1.0
                V[idx_v, col] = 1.0
                W[idx_w, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
