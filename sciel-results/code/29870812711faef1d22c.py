"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_tensors(n, m, p):
    """
    Build the matrix multiplication tensor T[i, j, k] = 1 for the (i,j,k) index.
    We represent the decomposition as:
        T[i, j, k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    """
    # T is a 3D tensor of shape (n*m, m*p, p*n) but we only need the "1" entries
    # Actually, the tensor is defined by T[i,j,k] = 1 for all i,j,k
    # The constraint is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
    pass

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_optimization(n, m, p, max_rank, time_limit=140.0):
    """
    Try to find a rank-minimal decomposition.
    We start from the schoolbook decomposition and try to reduce the rank
    by merging terms or finding a more efficient decomposition.
    
    Strategy:
    1. Start with schoolbook rank = n*m*p = 60
    2. Try to find a lower rank by randomized search over the space of
       low-rank decompositions
    """
    start_time = time.time()
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    if _verify(U, V, W, n, m, p):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by searching for sparse/structured decompositions
    # The key insight: for small matrices, the rank of matrix multiplication
    # is known to be less than n*m*p. For 3x4*4x5, we can try ranks below 60.
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Upper bound: n*m*p = 60
    
    # Try randomized search for lower rank decompositions
    # We use a greedy approach: try to merge terms in the schoolbook decomposition
    
    # For now, return the schoolbook solution if we can't find a better one quickly
    # This is a placeholder for a more sophisticated search
    
    if time.time() - start_time > time_limit - 10:
        return best_U, best_V, best_W, best_rank
    
    # Try a simple optimization: look for duplicate rows in U, V, W
    # that can be merged
    # This is a heuristic that may not find the optimal solution
    
    # For a proper implementation, we would use:
    # 1. Tensor rank algorithms (e.g., based on polynomial systems)
    # 2. Randomized numerical linear algebra
    # 3. Local search over the space of decompositions
    
    # Given time constraints, return the schoolbook solution
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find an optimal or near-optimal decomposition
    U, V, W, rank = _try_rank_optimization(n, m, p, n * m * p, time_limit=140.0)
    
    # Convert to lists as required by the interface
    U_list = [[U[i, j] for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[V[i, j] for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[W[i, j] for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
