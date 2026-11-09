import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a lower-rank exact tensor decomposition using
    randomized gradient-based optimization with rank reduction.
    Falls back to schoolbook if no valid lower-rank solution is found.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # where 0<=i<n, 0<=j<m, 0<=k<p
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Start with schoolbook rank
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def compute_error(U, V, W):
        """Compute Frobenius norm of reconstruction error."""
        # Reconstruct tensor: T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(target - T_approx)
    
    def is_valid(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        return compute_error(U, V, W) < tol
    
    # Try to reduce rank from schoolbook down
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Start with rank slightly below schoolbook and try to optimize
    for target_rank in range(schoolbook_rank - 1, max(1, schoolbook_rank // 2), -1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 5
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random small perturbations
            scale = 0.1
            U = U_sb[:, :target_rank] + np.random.randn(n * m, target_rank) * scale
            V = V_sb[:, :target_rank] + np.random.randn(m * p, target_rank) * scale
            W = W_sb[:, :target_rank] + np.random.randn(p * n, target_rank) * scale
            
            # Gradient-based optimization
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute error
                err = compute_error(U, V, W)
                if err < 1e-10:
                    found_valid = True
                    break
                
                # Compute gradients
                # d/dU: sum_{b,c} target[a,b,c] * V[b,r] * W[c,r] - sum_{b,c} U[a,r]*V[b,r]*W[c,r]*V[b,r]*W[c,r]
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = target - T_approx
                
                # Gradient for U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
                # Gradient for V[b,r] = sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
                grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
                # Gradient for W[c,r] = sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if found_valid:
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if found_valid:
            # Check if we can go lower
            continue
        else:
            # If we can't achieve this rank, stop trying lower ranks
            # (assuming monotonicity: if rank r fails, lower ranks likely fail too)
            # But let's try a few more ranks just in case
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
