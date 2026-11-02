import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank is n*m*p = 63
    rank_trivial = n * m * p
    
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m + j, col] = 1
                V_trivial[j*p + k, col] = 1
                W_trivial[k*n + i, col] = 1
                col += 1
    
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    best_rank = rank_trivial
    
    # Search for lower rank decompositions
    # Strategy: Try ranks from 30 up to 62
    for target_rank in range(30, min(63, rank_trivial)):
        if time.time() > deadline - 10:
            break
        
        # Initialize U with structured pattern
        U = np.zeros((n*m, target_rank))
        # Fill U with a structured pattern based on i and j
        for r in range(target_rank):
            # Create diverse patterns for each rank component
            i_idx = r % n
            j_idx = (r // n) % m
            U[i_idx*m + j_idx, r] = 1.0
            # Add some additional structure
            if r >= n*m:
                extra_i = (r // (n*m)) % n
                extra_j = (r // (n*m) // m) % m
                U[extra_i*m + extra_j, r] += 0.5
        
        # For each U, try to find V and W that satisfy the tensor equation
        # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # This is a system of equations. We can use iterative refinement.
        # Start with random initialization of V and W
        np.random.seed(42)
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Iteratively refine V and W
        converged = False
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    # Build the linear system for V[j*p+k, :]
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    # Solve A @ V[j*p+k, :] = ones(n)
                    try:
                        V_new, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = V_new
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = W_new
                    except:
                        pass
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        error += abs(val - 1.0)
            
            if error < 1e-8:
                converged = True
                break
        
        if converged and target_rank < best_rank:
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break  # Found a valid decomposition, stop searching for lower ranks
    
    # Verify the best result
    if best_rank < rank_trivial:
        U_best = np.array(best_result["U"])
        V_best = np.array(best_result["V"])
        W_best = np.array(best_result["W"])
        
        # Final verification
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_best[i*m+j, :] * V_best[j*p+k, :] * W_best[k*n+i, :])
                    error += abs(val - 1.0)
        
        if error < 1e-6:
            return best_result
    
    # Return trivial decomposition if no better found
    return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
