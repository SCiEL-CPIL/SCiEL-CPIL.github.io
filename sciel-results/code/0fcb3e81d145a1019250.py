"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    """
    Find a low-rank tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Uses randomized ALS-like iterative refinement to search for lower-rank decompositions.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to leave margin before 150s kill
    
    # Start with schoolbook as baseline
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using randomized ALS-like approach
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want to express T as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try ranks from 90 down to 50
    for target_rank in range(90, 49, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                    break
                    
            np.random.seed(trial * 1000 + target_rank)
            
            # Initialize with random values
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Iteratively refine using ALS-like approach
            # For each c, we want U[:,c] * V[:,c] * W[:,c] to approximate T
            # This is a non-convex problem, so we use iterative updates
            
            for iteration in range(50):
                if time.time() > deadline - 3:
                    break
                    
                # Update U
                for c in range(target_rank):
                    # For each (i,j), compute the residual
                    for i in range(n):
                        for j in range(m):
                            idx = i * m + j
                            # The contribution of other factors to T[i,j,k]
                            # T[i,j,k] = sum_c' U[idx, c'] * V[j*p+k, c'] * W[k*n+i, c']
                            # We want to find U[idx, c] such that this holds
                            # This is complex, so we use a simplified update
                            pass
                
                # Simplified: use gradient descent on the loss
                # Loss = sum_{i,j,k} (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
                
                # Compute gradients
                grad_U = np.zeros_like(U_rand)
                grad_V = np.zeros_like(V_rand)
                grad_W = np.zeros_like(W_rand)
                
                lr = 0.01 / (iteration + 1)
                
                # Sample a subset of (i,j,k) for speed
                num_samples = 1000
                for _ in range(num_samples):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Compute current value
                    current = 0.0
                    for c in range(target_rank):
                        current += U_rand[u_idx, c] * V_rand[v_idx, c] * W_rand[w_idx, c]
                    
                    error = current - 1.0
                    
                    # Gradients
                    for c in range(target_rank):
                        grad_U[u_idx, c] += error * V_rand[v_idx, c] * W_rand[w_idx, c]
                        grad_V[v_idx, c] += error * U_rand[u_idx, c] * W_rand[w_idx, c]
                        grad_W[w_idx, c] += error * U_rand[u_idx, c] * V_rand[v_idx, c]
                
                # Update
                U_rand -= lr * grad_U
                V_rand -= lr * grad_V
                W_rand -= lr * grad_W
                
                # Check if we've converged to a valid decomposition
                if iteration % 10 == 0:
                    # Convert to list format for verification
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
            else:
                continue
            break
        
        # If we found a valid decomposition at this rank, stop
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
