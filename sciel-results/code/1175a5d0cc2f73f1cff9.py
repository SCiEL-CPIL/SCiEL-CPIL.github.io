import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Pre-compute target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1
                
    # Reshape T_target to (pn, nm * mp) for easier solving
    # T_target[gamma, alpha, beta]
    T_flat = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    start_time = time.time()
    deadline = start_time + 130 # 130s limit to be safe
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 49 down to 35
    for R in range(49, 34, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 5000 # High attempts allowed due to fast check
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random integers -1, 0, 1
            # Using 0,1 might be better for sparsity, -1,1 for invertibility
            # Let's try a mix.
            U = np.random.randint(-1, 2, (nm, R))
            V = np.random.randint(-1, 2, (mp, R))
            
            # Construct A matrix of shape (nm * mp, R)
            # A[idx, r] = U[alpha, r] * V[beta, r]
            # alpha is 0..nm-1, beta is 0..mp-1
            # idx = alpha * mp + beta
            # We can do this via broadcasting
            # U is (nm, R), V is (mp, R)
            # We want (nm, mp, R)
            A_3d = U[:, None, :] * V[None, :, :]
            A = A_3d.reshape(nm * mp, R)
            
            # Solve for W for each gamma
            W_list = []
            valid = True
            for gamma in range(pn):
                t_gamma = T_flat[gamma]
                # Solve A W_g = t_gamma
                # A is (250, R), t_gamma is (250,)
                try:
                    W_g, residuals, rank, s = np.linalg.lstsq(A, t_gamma, rcond=None)
                    # Check residual
                    if residuals and residuals[0] > 1e-9:
                        valid = False
                        break
                    # Check if W_g is close to integer or half-integer
                    # Multiply by 2 to check integers
                    W_g_scaled = W_g * 2
                    W_g_rounded = np.round(W_g_scaled)
                    if np.max(np.abs(W_g_scaled - W_g_rounded)) > 1e-6:
                        valid = False
                        break
                    W_list.append(W_g_rounded / 2.0)
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if valid:
                # Assemble W
                W = np.vstack(W_list) # Shape (pn, R)
                
                # Verify exactly with integers
                # Convert to float for verification to avoid integer overflow issues if any, 
                # but entries are small.
                U_float = U.astype(np.float64)
                V_float = V.astype(np.float64)
                W_float = W.astype(np.float64)
                
                # Compute T_calc
                # T_calc[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                T_calc = np.einsum('ar,br,cr->abc', U_float, V_float, W_float)
                
                if np.array_equal(T_calc, T_target):
                    found = True
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break # Found a solution for this R, try lower R?
                    # The loop goes 49, 48...
                    # If we find 49, we break inner loop.
                    # Outer loop continues to 48.
                    # This is correct.
        
        # If we found a solution for R, we continue to R-1.
        # If we don't, we continue to R-1.
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
