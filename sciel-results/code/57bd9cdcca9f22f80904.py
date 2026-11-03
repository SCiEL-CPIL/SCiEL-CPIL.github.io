"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of the error with respect to U, V, W."""
    rank = U.shape[1]
    # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want S[i,j,k] = 1
    
    # Efficient computation of S
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    # We can compute the error and gradients in a loop over c or vectorized.
    # Vectorization is tricky due to the index permutations.
    # Let's use a loop over the small dimension or just iterate over indices if R is small.
    # R is at most 128. n*m*p = 128.
    # 128 * 128 operations is trivial.
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Initialize S to 0
    # We need to accumulate S[i,j,k]
    # To avoid a 4D array of size 4*4*8=128, we can just compute the error directly.
    
    # For each c, the contribution to S is U_c * V_c * W_c (with index mapping).
    # Let's precompute the index mappings.
    
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We will iterate over i, j, k to compute the error and gradients.
    # This is O(n*m*p*rank). 128 * 128 = 16k ops. Very fast.
    
    errors = np.zeros((n, m, p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i*m + j, :]
                vk = V[j*p + k, :]
                wk = W[k*n + i, :]
                s_val = np.dot(ui, vk * wk)
                errors[i, j, k] = s_val - 1.0
                
                # Gradients
                # dE/du = 2 * (s-1) * vk * wk
                # dE/dv = 2 * (s-1) * ui * wk
                # dE/dw = 2 * (s-1) * ui * vk
                
                factor = 2.0 * (s_val - 1.0)
                grad_U[i*m + j, :] += factor * (vk * wk)
                grad_V[j*p + k, :] += factor * (ui * wk)
                grad_W[k*n + i, :] += factor * (ui * vk)
                
    return errors, grad_U, grad_V, grad_W

def _optimize_decomposition(U, V, W, n, m, p, target_rank, max_iters=500, lr=0.05, seed=0):
    """Optimize U, V, W to minimize error, potentially reducing effective rank."""
    np.random.seed(seed)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # If we start with rank > target_rank, we should zero out some columns randomly?
    # Or better, initialize with random rank target_rank.
    # But we want to use the structure.
    # Strategy: Start with schoolbook, but if current rank > target_rank, 
    # we can't just delete. We need to initialize a new set of size target_rank.
    
    # For now, this function assumes U,V,W already have shape (..., target_rank).
    # It just optimizes the values.
    
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    beta = 0.9
    
    best_err = float('inf')
    
    for it in range(max_iters):
        errors, gU, gV, gW = _compute_gradients(U, V, W, n, m, p)
        err_norm = np.linalg.norm(errors)
        
        if err_norm < 1e-8:
            break
            
        # Update momentum
        momentum_U = beta * momentum_U + (1 - beta) * gU
        momentum_V = beta * momentum_V + (1 - beta) * gV
        momentum_W = beta * momentum_W + (1 - beta) * gW
        
        # Update parameters
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks
    # We will try a set of target ranks
    # Known lower bounds suggest we can go below 128.
    # Let's try 120, 110, 100, 98, 90, 80, 70, 60
    
    target_ranks = [120, 110, 100, 98, 90, 80, 70, 60]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try a few random initializations for this rank
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W with random values of rank R
            # Scaling is important.
            # If we use random values, they need to be small enough to not diverge,
            # but large enough to contribute.
            # A good init is to take the first R columns of schoolbook? No, that's biased.
            # Let's use random normal with small std.
            # Or, better: start with schoolbook, but project to rank R?
            # Actually, random init often works if lr is tuned.
            
            # Let's try initializing with the first R columns of the schoolbook decomposition
            # This provides a "structured" starting point close to the solution.
            # However, simply truncating might leave large errors in the "missing" parts.
            # Optimization should fix it.
            
            U_init = U_school[:, :R].copy()
            V_init = V_school[:, :R].copy()
            W_init = W_school[:, :R].copy()
            
            # Add small noise to break symmetry
            U_init += np.random.randn(*U_init.shape) * 0.01
            V_init += np.random.randn(*V_init.shape) * 0.01
            W_init += np.random.randn(*W_init.shape) * 0.01
            
            # Optimize
            U_opt, V_opt, W_opt, err = _optimize_decomposition(
                U_init, V_init, W_init, n, m, p, R, 
                max_iters=300, lr=0.01, seed=seed
            )
            
            if err < 1e-6:
                # Verify exactly
                # Re-check with high precision
                U_check = np.array(U_opt)
                V_check = np.array(V_opt)
                W_check = np.array(W_opt)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_check[i*m+j] * V_check[j*p+k] * W_check[k*n+i])
                            if abs(s - 1.0) > 1e-5:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                    
                if valid:
                    best_rank = R
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    print(f"Found valid decomposition with rank {R}")
                    break # Move to next lower rank
            
    # If we didn't find anything better, return schoolbook
    # But wait, we might have found a rank slightly lower than 128 in the loop above.
    # The loop goes 120, 110... so if 120 works, it's stored.
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
