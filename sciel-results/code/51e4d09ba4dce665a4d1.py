"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 48
    
    def make_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find better decomposition using tensor network approaches
    # The multiplication tensor T has T[i,j,k*n+i] = 1 for 3x4 matrix * 4x4 matrix
    
    # Let's try a simple approach: look for rank 47 by checking if any 
    # schoolbook term can be expressed as a combination of others
    # This is hard in general, so let's try a greedy approach
    
    # Build the multiplication tensor as a 3D array
    T = np.zeros((n, m, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k * n + i] = 1.0
    
    # Try different ranks using a simple optimization approach
    # We'll use the fact that for small matrices, known results exist
    # For 3x4*4x4, the minimal rank is likely close to 48
    
    # Let's try to verify if rank 47 is possible by checking linear dependencies
    # among the 48 rank-1 terms
    
    # Actually, let's try a practical approach: use ALS for a few iterations
    # to see if we can converge to a lower rank solution
    
    def try_rank(r, max_iters=1000, tol=1e-10):
        if time.time() > deadline:
            return None
        
        # Initialize with schoolbook-like structure
        np.random.seed(42)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Initialize with first r schoolbook terms
        for c in range(min(r, rank_schoolbook)):
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
        # If r < rank_schoolbook, we need a different initialization
        if r < rank_schoolbook:
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for c in range(r):
                # Residual in U direction
                VtW = V[:, c] @ W[:, c].T  # (m*p, p*n)
                # Target in U direction
                target = T.reshape(n, m, p * n).T  # (m, n, p*n)
                # For each (i, k*n+i) pair, we need contribution
                # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k*n+i]
                
                # Let's do proper ALS: fix V[:,c] and W[:,c], solve for U[:,c]
                # Sum over j,k: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k*n+i]
                # This is a linear system for U[:,c]
                
                # Build the linear system
                # For each row i*m+j of U:
                # Sum_k U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k*n+i] for all i,k
                # Group by j:
                # For fixed j, Sum_k U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k*n+i]
                # This is a coupled system
                
                # Let's just do coordinate descent instead
                for idx in range(n * m):
                    i = idx // m
                    j = idx % m
                    # Gradient for U[idx, c]
                    grad = 0.0
                    obj = 0.0
                    for k in range(p):
                        target_val = T[i, j, k * n + i]
                        pred_val = U[idx, c] * V[j * p + k, c] * W[k * n + i, c]
                        obj += (pred_val - target_val) ** 2
                        grad += 2 * (pred_val - target_val) * V[j * p + k, c] * W[k * n + i, c]
                    
                    if abs(grad) > 1e-15:
                        # Simple gradient step
                        lr = 1e-4
                        U[idx, c] -= lr * grad
                    
                    # Check convergence
                    if abs(grad) < tol:
                        pass
                
                if time.time() > deadline:
                    return None
            
            # Update V similarly
            for c in range(r):
                for idx in range(m * p):
                    j = idx // p
                    k = idx % p
                    grad = 0.0
                    for i in range(n):
                        target_val = T[i, j, k * n + i]
                        pred_val = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        grad += 2 * (pred_val - target_val) * U[i * m + j, c] * W[k * n + i, c]
                    
                    if abs(grad) > 1e-15:
                        lr = 1e-4
                        V[idx, c] -= lr * grad
            
            # Update W similarly
            for c in range(r):
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    grad = 0.0
                    for j in range(m):
                        target_val = T[i, j, k * n + i]
                        pred_val = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        grad += 2 * (pred_val - target_val) * U[i * m + j, c] * V[j * p + k, c]
                    
                    if abs(grad) > 1e-15:
                        lr = 1e-4
                        W[idx, c] -= lr * grad
            
            # Check convergence
            if it % 100 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target_val = T[i, j, k * n + i]
                            pred_val = 0.0
                            for c in range(r):
                                pred_val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (pred_val - target_val) ** 2
                if err < 1e-8:
                    # Convert to list format
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    # Try ranks from 47 down to 40
    for r in range(47, 39, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=200)
        if result is not None:
            # Verify
            U_arr = np.array(result["U"])
            V_arr = np.array(result["V"])
            W_arr = np.array(result["W"])
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target_val = T[i, j, k * n + i]
                        pred_val = 0.0
                        for c in range(r):
                            pred_val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                        if abs(pred_val - target_val) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return result
    
    # Fallback to schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END

