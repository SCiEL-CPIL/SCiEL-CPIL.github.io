"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct Schoolbook Decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
                
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomposition(U, V, W, rank):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
        """
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def compute_error(U, V, W, rank):
        """Compute the maximum absolute error from the target tensor (all ones)."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(T - 1.0))

    # Try to find lower rank decompositions by structured optimization
    # Strategy: for rank r < 120, take first r columns of schoolbook and optimize
    # to minimize error. If error reaches ~0, we have a valid decomposition.
    
    for target_rank in range(119, 100, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            # Initialize with first target_rank columns of schoolbook
            U_init = U_school[:, :target_rank].copy()
            V_init = V_school[:, :target_rank].copy()
            W_init = W_school[:, :target_rank].copy()
            
            # Add small random perturbation to escape local minima
            if attempt > 0:
                noise_scale = 0.1 * attempt
                U_init += np.random.normal(0, noise_scale, U_init.shape)
                V_init += np.random.normal(0, noise_scale, V_init.shape)
                W_init += np.random.normal(0, noise_scale, W_init.shape)
            
            # Use gradient-free optimization: Nelder-Mead style or simple coordinate descent
            # For speed, use a simple iterative approach
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Try a few optimization steps
            # We'll use a simple approach: for each column, try to adjust it
            # to reduce the error
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline - 30:
                    break
                
                err = compute_error(U_curr, V_curr, W_curr, target_rank)
                if err < 1e-9:
                    found = True
                    break
                
                # Compute gradients approximately
                # For each entry in U, V, W, compute the partial derivative of the error
                # This is expensive, so use a simpler approach: random coordinate updates
                
                # Pick a random matrix and a random entry
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, target_rank-1)
                    old_val = U_curr[row, col]
                    # Try a small perturbation
                    delta = 0.01 * (random.random() - 0.5)
                    U_curr[row, col] = old_val + delta
                    new_err = compute_error(U_curr, V_curr, W_curr, target_rank)
                    if new_err > err:
                        U_curr[row, col] = old_val  # Revert
                    else:
                        # Keep and try to go further
                        pass
                elif mat_choice == 1:
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, target_rank-1)
                    old_val = V_curr[row, col]
                    delta = 0.01 * (random.random() - 0.5)
                    V_curr[row, col] = old_val + delta
                    new_err = compute_error(U_curr, V_curr, W_curr, target_rank)
                    if new_err > err:
                        V_curr[row, col] = old_val
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, target_rank-1)
                    old_val = W_curr[row, col]
                    delta = 0.01 * (random.random() - 0.5)
                    W_curr[row, col] = old_val + delta
                    new_err = compute_error(U_curr, V_curr, W_curr, target_rank)
                    if new_err > err:
                        W_curr[row, col] = old_val
            
            if found:
                # Verify exactly
                if verify_decomposition(U_curr, V_curr, W_curr, target_rank):
                    best_rank = target_rank
                    best_U = U_curr.copy()
                    best_V = V_curr.copy()
                    best_W = W_curr.copy()
                    break
                else:
                    # Check if close enough and round to half-integers
                    U_rounded = np.round(U_curr * 2) / 2
                    V_rounded = np.round(V_curr * 2) / 2
                    W_rounded = np.round(W_curr * 2) / 2
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, target_rank):
                        best_rank = target_rank
                        best_U = U_rounded.copy()
                        best_V = V_rounded.copy()
                        best_W = W_rounded.copy()
                        found = True
                        break
        
        if found:
            break
    
    # Final verification of the best result
    if not verify_decomposition(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook
        best_rank = rank_schoolbook
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
