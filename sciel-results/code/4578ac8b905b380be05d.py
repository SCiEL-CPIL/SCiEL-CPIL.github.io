import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    start = time.time()
    deadline = 15.0  # Safety margin
    
    rank = n * m * p  # 45
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill tensors according to schoolbook algorithm
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    
    # Try to find lower-rank decomposition using numpy
    # First, verify the tensor is exact
    # Then try to find a lower-rank exact decomposition
    
    # Convert to numpy arrays
    U_arr = np.array(U)  # shape (n*m, rank)
    V_arr = np.array(V)  # shape (m*p, rank)
    W_arr = np.array(W)  # shape (p*n, rank)
    
    # The multiplication tensor T[i*m + j, k*n + i, j*p + k] = 1
    # In matrix form: for each rank term r, the contribution is
    # U[:, r] @ (something) + V[:, r] @ (something) + W[:, r] @ (something)
    
    # Actually, let's verify the tensor decomposition directly
    # Build the 3-tensor T from the matrices
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += U_arr[:, r, np.newaxis, np.newaxis] * V_arr[:, np.newaxis, r, np.newaxis] * W_arr[:, np.newaxis, np.newaxis, r]
    
    # T should have T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Let's verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = T[i*m+j, j*p+k, k*n+i]
                if abs(val - 1.0) > 1e-9:
                    return {"U": U, "V": V, "W": W}
    
    # Now try to find a lower-rank exact decomposition
    # This is a hard problem. Let's try a few approaches
    
    # Approach: Try rank 44, 43, ... by removing rank terms and seeing if
    # the tensor can be exactly represented with fewer terms
    
    # Since this is NP-hard in general, we'll try a simple heuristic:
    # For each pair of rank terms, check if removing both allows exact
    # representation with the remaining terms plus a correction
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
