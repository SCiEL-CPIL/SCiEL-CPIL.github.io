"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find a lower-rank decomposition using random initialization and refinement
    # The tensor for matrix multiplication M_{i,j,k} = A_{i,j} * B_{j,k} -> C_{i,k}
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a 3-tensor decomposition problem.
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Upper bound: n*m*p = 90 (schoolbook)
    # Strassen-like algorithms might help, but for 3x5x6 it's not straightforward.
    # Let's try to find a rank slightly below 90 using optimization.
    
    # Use a simple approach: start with schoolbook and try to merge terms
    # Or use random initialization with gradient-free optimization
    
    # For this specific case, let's try ranks from 30 up to 89
    # But we need to be efficient. Let's try a few random initializations for lower ranks.
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute the tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Check if all entries are 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def try_rank(rank):
        """Try to find a valid decomposition with given rank."""
        if rank >= best_rank:
            return None
        
        # Random initialization
        # We need U (n*m, rank), V (m*p, rank), W (p*n, rank)
        # such that for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Strategy: use random vectors and check
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Try a few iterations of a simple update rule
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Compute errors
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = s - 1.0
                            max_err = max(max_err, abs(err))
                
                if max_err < 1e-8:
                    # Found a solution!
                    # Convert to list of lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Simple gradient step (approximate)
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = s - 1.0
                            # Update U
                            grad_U = err * V[j*p+k, :] * W[k*n+i, :]
                            U[i*m+j, :] -= lr * grad_U
                            # Update V
                            grad_V = err * U[i*m+j, :] * W[k*n+i, :]
                            V[j*p+k, :] -= lr * grad_V
                            # Update W
                            grad_W = err * U[i*m+j, :] * V[j*p+k, :]
                            W[k*n+i, :] -= lr * grad_W
            
            # Check final state
            if validate(U.tolist(), V.tolist(), W.tolist()):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try a few lower ranks
    for rank in [85, 80, 75, 70, 65, 60]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank)
        if result is not None:
            # Validate
            if validate(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
                # Can we go lower?
                continue
        else:
            break  # If this rank failed, higher ranks (lower values) might also fail
    
    return best_result
# EVOLVE-BLOCK-END
