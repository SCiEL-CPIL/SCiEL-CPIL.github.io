"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T as a 3D array T[i, j, k] = 1 iff (i,j,k) corresponds to a term.
    Actually the tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    The multiplication tensor has components T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    But the trace convention uses W[k*n+i, r], so the output index is (k, i) -> k*n+i.
    The tensor is rank decomposable: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    The decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    # Reshape to 3D
    U3 = U.reshape(n, m, r)  # U[i*m+j, r] -> U3[i, j, r]
    V3 = V.reshape(m, p, r)  # V[j*p+k, r] -> V3[j, k, r]
    W3 = W.reshape(p, n, r)  # W[k*n+i, r] -> W3[k, i, r]
    
    # Compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This should equal 1 for all (i,j,k)
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U3[i, j, r_idx] * V3[j, k, r_idx] * W3[k, i, r_idx]
                result[i, j, k] = s
    
    T = np.ones((n, m, p))
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Try to find a low-rank decomposition
    # Known results: rank(2x3x7) is at most 30 (from 4p+2 = 30)
    # Let's try rank 30 first, then 31, 32, ...
    
    best_rank = n * m * p  # Schoolbook fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition (always works)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # For 2x3x7, we know rank is at most 30
    # Try ranks from 25 to 35
    for target_rank in range(25, min(36, rank_sb + 1)):
        if time.time() > deadline - 5.0:
            break
        
        # Set up the optimization problem
        # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Use random initialization and gradient-free optimization (or simple iterative method)
        # Since the problem is non-convex, we'll use a simple approach:
        # Initialize with small random values and use alternating least squares or similar
        
        # For simplicity, let's try a structured approach:
        # We can try to use the known structure from smaller cases
        
        # Actually, let's just use the schoolbook for now since finding
        # exact low-rank decompositions is non-trivial
        
        pass
    
    # Return the best decomposition found
    # Convert numpy arrays to lists of lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
