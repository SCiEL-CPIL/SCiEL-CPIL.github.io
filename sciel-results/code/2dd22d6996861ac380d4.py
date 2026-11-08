"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
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
    
    # Build the target tensor T[i*m+j, j*p+k] = 1 if the indices match appropriately
    # T[a, b] where a = i*m+j, b = j*p+k, and T = sum_c U[a,c]*V[b,c]*W[k*n+i, c]
    # Actually the tensor is T_{i,j,k} = A_{ij} * B_{jk}
    # In the decomposition: sum_c U_{ij,c} * V_{jk,c} * W_{ki,c} = 1 for all i,j,k
    
    # Let's use a randomized approach to find a lower rank decomposition
    # Try various ranks and use least squares to fit
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a decomposition with lower rank using numerical optimization
    # The constraint is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a hard problem. Let's try a structured approach.
    # Use the fact that we can sometimes combine terms.
    
    # Try rank values that are divisors or related to n*m*p
    candidate_ranks = [120, 110, 100, 96, 90, 84, 80, 78, 72, 70, 66, 60]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        # Use a simple iterative approach: initialize randomly, then project
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Iterate to satisfy constraints
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of n*m*p equations
        
        # Use a simple gradient-free approach: for each equation, adjust one variable
        # This is very slow, so let's limit iterations
        
        max_iters = 50
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Check current error
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        errors.append(s - 1.0)
            errors = np.array(errors)
            max_err = np.max(np.abs(errors))
            
            if max_err < 1e-6:
                # Found a solution, try to round to integers/half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify the rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to lists
                    U_list = [[U_int[i, c] for c in range(target_rank)] for i in range(n * m)]
                    V_list = [[V_int[i, c] for c in range(target_rank)] for i in range(m * p)]
                    W_list = [[W_int[i, c] for c in range(target_rank)] for i in range(p * n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
            
            # Simple update: adjust one random variable to reduce error
            idx = np.random.randint(len(errors))
            # Decode idx to (i, j, k)
            idx_flat = idx
            k = idx_flat % p
            idx_flat //= p
            j = idx_flat % m
            idx_flat //= m
            i = idx_flat
            
            err = errors[idx]
            if abs(err) < 1e-10:
                continue
            
            # Adjust U[i*m+j, :] in direction of V[j*p+k,:]*W[k*n+i,:]
            grad = V[j*p+k, :] * W[k*n+i, :]
            if np.linalg.norm(grad) < 1e-10:
                continue
            step = err / (np.dot(grad, grad) + 1e-10)
            U[i*m+j, :] -= step * grad
        
        if time.time() > deadline:
            break
    
    return best_result
# EVOLVE-BLOCK-END
