"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Uses a local search / optimization approach to find low-rank decompositions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(r):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def to_half_integer_list(arr):
        """Convert numpy array to list of lists with half-integer precision."""
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    def schoolbook():
        """Return the schoolbook rank-27 decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a low-rank decomposition using iterative optimization
    # Start from rank 23 and try to find a valid decomposition
    
    best_result = None
    best_rank = n * m * p
    
    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook()
    if verify(U_sb, V_sb, W_sb):
        best_result = (U_sb, V_sb, W_sb)
        best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use a randomized initialization with local optimization
    
    for target_rank in range(23, best_rank):  # Try ranks 23, 24, ..., best_rank-1
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_tries = 30
        for trial in range(max_tries):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Use iterative refinement: for each rank-1 term, try to adjust
            # to better fit the tensor
            
            # Define the tensor T[i,j,k] = 1 for all i,j,k
            # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            
            # Use a simple iterative approach: alternate fixing two factors and
            # solving for the third
            
            success = False
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # For each column c, try to improve the fit
                # Compute current residual
                residual = np.zeros((n, m, p))
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                # Check if we're close
                max_err = np.max(np.abs(residual - 1.0))
                if max_err < 1e-9:
                    success = True
                    break
                
                # Update W: for each (k,i) and each c, we want
                # W[k*n+i,c] = (1 - sum_{c'≠c} U[i*m+j,c']*V[j*p+k,c']) / U[i*m+j,c] * V[j*p+k,c]
                # This is tricky because it depends on j. Let's use a different approach.
                
                # Use gradient descent on the loss L = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                # dL/dW[k*n+i,c] = 2 * (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1) * U[i*m+j,c] * V[j*p+k,c]
                # But this sums over j, which is problematic.
                
                # Alternative: use a simpler local search - perturb random entries
                # and accept if it improves the max error
                
                # Perturb a random entry
                factor = np.random.randint(3)
                if factor == 0:
                    idx = np.random.randint(n*m)
                    c = np.random.randint(target_rank)
                    old_val = U[idx, c]
                    U[idx, c] += np.random.randn() * 0.1
                    # Check improvement
                    new_residual = np.zeros((n, m, p))
                    for c2 in range(target_rank):
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    new_residual[i2,j2,k2] += U[i2*m+j2,c2] * V[j2*p+k2,c2] * W[k2*n+i2,c2]
                    new_max_err = np.max(np.abs(new_residual - 1.0))
                    if new_max_err > max_err:
                        U[idx, c] = old_val
                elif factor == 1:
                    idx = np.random.randint(m*p)
                    c = np.random.randint(target_rank)
                    old_val = V[idx, c]
                    V[idx, c] += np.random.randn() * 0.1
                    new_residual = np.zeros((n, m, p))
                    for c2 in range(target_rank):
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    new_residual[i2,j2,k2] += U[i2*m+j2,c2] * V[j2*p+k2,c2] * W[k2*n+i2,c2]
                    new_max_err = np.max(np.abs(new_residual - 1.0))
                    if new_max_err > max_err:
                        V[idx, c] = old_val
                else:
                    idx = np.random.randint(p*n)
                    c = np.random.randint(target_rank)
                    old_val = W[idx, c]
                    W[idx, c] += np.random.randn() * 0.1
                    new_residual = np.zeros((n, m, p))
                    for c2 in range(target_rank):
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    new_residual[i2,j2,k2] += U[i2*m+j2,c2] * V[j2*p+k2,c2] * W[k2*n+i2,c2]
                    new_max_err = np.max(np.abs(new_residual - 1.0))
                    if new_max_err > max_err:
                        W[idx, c] = old_val
            
            if success:
                if verify(U, V, W):
                    best_result = (U, V, W)
                    best_rank = target_rank
                    break
        
        if best_rank <= target_rank:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_half_integer_list(U), "V": to_half_integer_list(V), "W": to_half_integer_list(W)}
    
    # Fallback to schoolbook
    return {"U": to_half_integer_list(U_sb), "V": to_half_integer_list(V_sb), "W": to_half_integer_list(W_sb)}
# EVOLVE-BLOCK-END
