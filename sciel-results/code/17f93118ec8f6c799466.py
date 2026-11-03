"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W uses trace convention: W[k*n+i, r]
    
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook decomposition as baseline
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Precompute index mappings for tensor evaluation
    # Tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Build list of (i,j,k) tuples and corresponding flat indices
    ijks = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ijks.append((i, j, k))
    
    num_terms = len(ijks)  # n*m*p = 45
    
    def compute_residual_vec(U, V, W):
        """Compute residual tensor and gradients using vectorized operations."""
        # Reshape U, V, W for einsum
        # U: (n*m, R) -> (n, m, R)
        # V: (m*p, R) -> (m, p, R)
        # W: (p*n, R) -> (p, n, R)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Residual: Res[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # P[i,j,k,c] = U[i,j,c] * V[j,k,c]
        P = np.einsum('ijc,jkc->ijkc', U_r, V_r)
        Res = np.einsum('ijkc,kic->ijk', P, W_r)
        
        # Gradients
        # GradU[i,j,c] = sum_{j,k} Res[i,j,k] V[j,k,c] W[k,i,c]
        # VW[i,j,c] = sum_k V[j,k,c] W[k,i,c]
        VW = np.einsum('jkc,kic->ijc', V_r, W_r)
        GradU = np.einsum('ijk,ijc->ijc', Res, VW)
        
        # GradV[j,k,c] = sum_{i,k} Res[i,j,k] U[i,j,c] W[k,i,c]
        # UW[j,k,c] = sum_i U[i,j,c] W[k,i,c]
        UW = np.einsum('ijc,kic->jkc', U_r, W_r)
        GradV = np.einsum('ijk,jkc->jkc', Res, UW)
        
        # GradW[k,i,c] = sum_{i,j} Res[i,j,k] U[i,j,c] V[j,k,c]
        # UV[i,k,c] = sum_j U[i,j,c] V[j,k,c]
        UV = np.einsum('ijc,jkc->ikc', U_r, V_r)
        GradW = np.einsum('ijk,ikc->kic', Res, UV)
        
        # Flatten gradients back to original shapes
        grad_U = GradU.reshape(n*m, rank)
        grad_V = GradV.reshape(m*p, rank)
        grad_W = GradW.reshape(p*n, rank)
        
        max_res = np.max(np.abs(Res))
        return max_res, grad_U, grad_V, grad_W
    
    # Try to find lower rank decompositions using gradient-based optimization
    # For matrix multiplication, known lower bounds exist but let's search
    
    # The tensor equation is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from low to high
    # Known: rank is at most n*m*p = 45
    # For 3x3x3, rank is 23 (Strassen-like improvements)
    # For 3x3x5, let's try various ranks
    
    # Use a more systematic approach: try ranks and use gradient descent
    # Start from best_rank - 1 down to a reasonable limit
    # Since schoolbook is 45, we try 44, 43, ...
    # We limit the number of attempts and iterations to ensure speed
    max_attempts = 20
    max_iterations = 100
    
    for rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structured perturbation (most likely to work)
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Map schoolbook indices to this rank
            # If rank < n*m*p, we can't just copy schoolbook.
            # We use a random sparse init or small random.
            # Let's use small random for simplicity and speed.
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            lr = 0.05
            max_res, _, _, _ = compute_residual_vec(U, V, W)
            converged = False
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 2:
                    break
                
                max_res, grad_U, grad_V, grad_W = compute_residual_vec(U, V, W)
                
                if max_res < 1e-6:
                    converged = True
                    break
                
                # Simple gradient step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check if we improved, otherwise reduce lr
                new_res, _, _, _ = compute_residual_vec(U, V, W)
                if new_res > max_res:
                    lr *= 0.5
                    U += lr * grad_U
                    V += lr * grad_V
                    W += lr * grad_W
                else:
                    max_res = new_res
            
            if converged:
                # Verify with exact check
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                valid = True
                # Use vectorized check for speed
                max_res_check, _, _, _ = compute_residual_vec(U_rounded, V_rounded, W_rounded)
                if max_res_check < 1e-5:
                    valid = True
                else:
                    valid = False
                
                if valid:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break  # Found rank, try to go lower
        
        if found:
            continue
        else:
            # If we didn't find a solution for this rank, we can stop searching lower
            # because lower ranks are harder
            # But maybe skip a few? For now, break to save time.
            break
    
    # Convert to lists of lists with integer/half-integer entries
    def to_half_int_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                h = round(val * 2) / 2
                new_row.append(h)
            result.append(new_row)
        return result
    
    U_out = to_half_int_matrix(best_U)
    V_out = to_half_int_matrix(best_V)
    W_out = to_half_int_matrix(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
