"""Matrix multiplication 2x3 x 3x9 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor indices
    # For matrix mult: T[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    target_set = set(target_indices)
    
    # Structured decomposition: rank = m*p = 27
    # For each (j,k) pair, create a rank-1 term:
    #   U[i*m+j] = 1 for i in 0..n-1
    #   V[j*p+k] = 1
    #   W[k*n+i] = 1 for i in 0..n-1
    # This gives exact decomposition with integer entries.
    
    rank_struct = m * p  # 27
    U = np.zeros((dim_u, rank_struct))
    V = np.zeros((dim_v, rank_struct))
    W = np.zeros((dim_w, rank_struct))
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            for i in range(n):
                W[k * n + i, col] = 1
            col += 1
    
    # Verify the structured decomposition
    def verify_full(U, V, W, tol=1e-8):
        """Full verification of the decomposition."""
        # Check all target entries
        for (a, b, c) in target_indices:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        # Check all zero entries (full check since dimensions are small)
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    if (a, b, c) not in target_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > tol:
                            return False
        return True
    
    if verify_full(U, V, W, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition (rank 54)
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
