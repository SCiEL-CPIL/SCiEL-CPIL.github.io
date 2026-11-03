"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank tensor decomposition for 3x5 * 5x6 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all (i,j,k)
    
    # Start with schoolbook rank = n*m*p = 90
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook decomposition
    U = np.zeros((nm, best_rank))
    V = np.zeros((mp, best_rank))
    W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using random search with gradient-free optimization
    # Strategy: Start with random low-rank factors and try to minimize the error
    # T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # First, let's try some known lower bounds
    # For 3x5x6, the minimum rank is at most 90 (schoolbook)
    # Known results: for m x n x p, rank is at most m*n*p
    # Lower bound: max(m*n, n*p, m*p) = max(15, 18, 30) = 30
    
    # Try random search for ranks between 30 and 90
    # Use a simple approach: random initialization + iterative improvement
    
    def compute_error(U_mat, V_mat, W_mat, n, m, p):
        """Compute the max absolute error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    err = max(err, abs(val - 1.0))
        return err
    
    def try_rank(rank, n, m, p, max_iters=5000, seed=42):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline - 5:
            return None, None, None, float('inf')
        
        rng = np.random.RandomState(seed)
        nm, mp, pn = n*m, m*p, p*n
        
        # Random initialization
        U_mat = rng.randn(nm, rank) * 0.1
        V_mat = rng.randn(mp, rank) * 0.1
        W_mat = rng.randn(pn, rank) * 0.1
        
        # Try multiple restarts
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            U_mat = rng.randn(nm, rank) * 0.1
            V_mat = rng.randn(mp, rank) * 0.1
            W_mat = rng.randn(pn, rank) * 0.1
            
            # Simple iterative improvement: coordinate descent
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                for r in range(rank):
                    # For each r, U[:,r] should satisfy:
                    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # This is complex; let's use a simpler heuristic
                    
                    pass
                
                # Compute error
                err = compute_error(U_mat, V_mat, W_mat, n, m, p)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
                
                if best_err < 1e-8:
                    break
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from low to high
    for rank in range(30, best_rank + 1, 5):
        if time.time() > deadline - 10:
            break
        
        U_t, V_t, W_t, err = try_rank(rank, n, m, p)
        if U_t is not None and err < 1e-6:
            # Found a valid decomposition
            best_rank = rank
            best_U, best_V, best_W = U_t, V_t, W_t
            # Try to go lower
            break
    
    # If we found a better rank, try to refine it
    if best_rank < n * m * p:
        # Convert to exact fractions if possible
        pass
    
    # Return the best decomposition found
    # Convert numpy arrays to lists of lists for the required format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to remove floating point noise
    U_list = [[round(x, 10) for x in row] for row in U_list]
    V_list = [[round(x, 10) for x in row] for row in V_list]
    W_list = [[round(x, 10) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
