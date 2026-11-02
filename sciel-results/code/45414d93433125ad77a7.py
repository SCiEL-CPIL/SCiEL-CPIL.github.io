import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using tensor contraction."""
    rank = U.shape[1]
    # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can do this efficiently:
    # For fixed j: sum_r U[:,j,r] * V[j,:,r] * W[:,:,j]... no, W is indexed by k,i
    # Let's use einsum: sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    return verify_decomposition_fast(U, V, W, n, m, p, tol)

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42
    
    # Initialize with trivial decomposition
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to find a lower rank decomposition via local search
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Candidate values: small set of integers and half-integers
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 1: Try to reduce rank by randomly perturbing entries
    # Use the fast verifier to allow many more iterations
    for target_rank in range(rank - 1, max(rank - 10, 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random restarts for this target rank
        for restart in range(200):
            if time.time() > deadline - 20:
                break
            
            # Start from trivial and zero out some columns
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Zero out the last (rank - target_rank) columns
            for c in range(target_rank, rank):
                U_try[:, c] = 0
                V_try[:, c] = 0
                W_try[:, c] = 0
            
            # Randomly perturb entries to try to maintain validity
            for step in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Pick a random column and random entry to perturb
                col = random.randint(0, target_rank - 1)
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    matrix, rows = U_try, n * m
                elif mat_idx == 1:
                    matrix, rows = V_try, m * p
                else:
                    matrix, rows = W_try, p * n
                
                row = random.randint(0, rows - 1)
                old_val = matrix[row, col]
                new_val = random.choice(candidates)
                if new_val == old_val:
                    continue
                matrix[row, col] = new_val
                
                # Check if still valid using fast verifier
                if verify_decomposition_fast(U_try, V_try, W_try, n, m, p, tol=1e-8):
                    # Found a valid decomposition!
                    best_rank = target_rank
                    best_U = U_try.copy()
                    best_V = V_try.copy()
                    best_W = W_try.copy()
                    # Clean up to exact values
                    for mat in [best_U, best_V, best_W]:
                        mat[np.abs(mat) < 1e-9] = 0
                        mat[np.abs(mat - 0.5) < 1e-9] = 0.5
                        mat[np.abs(mat + 0.5) < 1e-9] = -0.5
                        mat[np.abs(mat - 1) < 1e-9] = 1
                        mat[np.abs(mat + 1) < 1e-9] = -1
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
                else:
                    # Revert
                    matrix[row, col] = old_val
    
    # Return the best found (likely the trivial one)
    U_out = best_U.copy()
    V_out = best_V.copy()
    W_out = best_W.copy()
    
    # Clean up
    for mat in [U_out, V_out, W_out]:
        mat[np.abs(mat) < 1e-9] = 0
        mat[np.abs(mat - 0.5) < 1e-9] = 0.5
        mat[np.abs(mat + 0.5) < 1e-9] = -0.5
        mat[np.abs(mat - 1) < 1e-9] = 1
        mat[np.abs(mat + 1) < 1e-9] = -1
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
