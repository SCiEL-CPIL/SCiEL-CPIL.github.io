"""Strategy: Numerical Optimization (Gradient Descent) for Tensor Rank Minimization.

The current seed program uses a fixed, hand-derived block decomposition (Rank 30 Schoolbook).
This new strategy re-frames the problem as a continuous optimization problem.
We initialize a random rank-r decomposition and use gradient descent to minimize the
squared error ||T_reconstructed - T_target||^2.
If we can drive the error to near-zero (within floating point tolerance), we have found
a valid rank-r decomposition. We try ranks in ascending order (14, 15, 16, 18) to find
the minimal rank achievable within the time budget.
"""

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 115  # Safety margin: 150s limit, leave 35s for safety/overhead
    
    # Target Tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # Dimensions: n=2, m=3, p=5
    # T shape: (n, m, p)
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    # We want to find U (n*m, r), V (m*p, r), W (p*n, r)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks in increasing order. 
    # Lower bound for 2x3 x 3x5 is not trivially 1, but likely > 10.
    # Schoolbook is 30. 
    # Let's try 12, 14, 16, 18, 20, 22, 24, 26, 28, 30.
    # If we find a valid one at rank R, we return it.
    
    candidate_ranks = [12, 14, 16, 18, 20, 24, 30]
    
    best_result = None
    best_rank = 30 # Fallback to schoolbook if all fail (should not happen)
    
    def verify_decomposition(U, V, W, tol=1e-5):
        """Check if the decomposition is exact enough."""
        err = 0.0
        # Efficient check: sum over r of outer products
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this via einsum or loops. Loops are safer for small sizes.
        # But for speed in optimization loop, we might just check norm.
        # For final verification, let's do a precise check.
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(U.shape[1]):
            # u_vec shape (n, m), v_vec shape (m, p), w_vec shape (p, n)
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # Add rank-1 tensor
            T_recon += u[:, :, np.newaxis] * v[np.newaxis, :, :] * w.T[np.newaxis, :, :] # Wait, W is (p*n, r). W[:, r] is (p*n,). 
            # W index is k*n + i. So w_vec[k*n+i].
            # w_matrix[k, i] = w_vec[k*n+i]
            # The term is u[i,j] * v[j,k] * w[k,i]
            # My loop above:
            # u[i,j] * v[j,k] * w.T[i,k]? No.
            # Let's just do the explicit sum for verification to be sure.
            pass 
        
        # Let's use a more direct verification method
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(U.shape[1]):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i*m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_col[j*p + k]
                        if v_val == 0: continue
                        w_val = w_col[k*n + i]
                        T_recon[i, j, k] += u_val * v_val * w_val
        
        diff = T_recon - T_target
        norm_diff = np.linalg.norm(diff)
        return norm_diff < tol, norm_diff

    def optimize_rank(r, time_limit=10.0):
        """Attempt to find a valid rank-r decomposition using gradient descent."""
        start_time = time.time()
        
        # Initialize with small random values
        # Scale matters. If values are too small, gradient is small. If too large, instability.
        # Target sum is 1. With r terms, average magnitude per term should be around 1/sqrt(r) or similar.
        # Let's init with uniform distribution in [-1, 1] or similar.
        scale = 1.0 / np.sqrt(r)
        
        U = np.random.uniform(-scale, scale, (n*m, r))
        V = np.random.uniform(-scale, scale, (m*p, r))
        W = np.random.uniform(-scale, scale, (p*n, r))
        
        # Learning rate
        lr = 0.01
        # We might need adaptive lr or multiple restarts if it fails.
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Simple Gradient Descent
        # Loss L = 0.5 * sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
        # dL/dU[i,j,r] = (sum_r' ... - 1) * V[j,k,r] * W[k,i,r] summed over k? No.
        # Let E[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
        # dL/dU[i,j,r] = sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
        # dL/dV[j,k,r] = sum_i E[i,j,k] * U[i,j,r] * W[k,i,r]
        # dL/dW[k,i,r] = sum_j E[i,j,k] * U[i,j,r] * V[j,k,r]
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute Error Tensor E
            E = np.zeros((n, m, p), dtype=np.float64)
            for r in range(r):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                # E[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                # Use broadcasting
                # u: (n, m, 1), v: (1, m, p), w: (p, n) -> need (n, m, p)
                # w[k,i] is w.T[i,k] if w is (p,n).
                # Let's stick to loops for gradient calculation to avoid shape bugs, 
                # but optimize the inner loop.
                # Actually, for n=2, m=3, p=5, the tensor is 30 elements.
                # r is small (12-30).
                # This is very fast.
                
                # Efficient update of E using einsum?
                # E += np.einsum('ij,jk,ki->ijk', u, v, w.T) 
                # Wait, w is (p, n). w[k, i]. So w.T is (n, p).
                # We need w[k, i] in the product u[i,j] v[j,k] w[k,i].
                # Let's define W_mat such that W_mat[i, k] = w[k, i]. That is w.T.
                E += np.einsum('ij,jk,ki->ijk', u, v, w.T)
            
            E -= 1.0 # Subtract target (which is 1)
            
            # Check convergence
            err = np.linalg.norm(E)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-6:
                return best_U, best_V, best_W, err
            
            # Compute Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(r):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                w_T = w.T # (n, p), w_T[i, k] = w[k, i]
                
                # dU[i,j,r] = sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
                # dU[:, :, r] = sum_k E[i,j,k] v[j,k] w[k,i]
                # = einsum('ijk,jk,ki->ij', E, v, w.T)
                dU[:, r] = np.einsum('ijk,jk,ki->ij', E, v, w.T).reshape(n*m)
                
                # dV[j,k,r] = sum_i E[i,j,k] * U[i,j,r] * W[k,i,r]
                # dV[:, :, r] = sum_i E[i,j,k] u[i,j] w[k,i]
                # = einsum('ijk,ij,ki->jk', E, u, w.T)
                dV[:, r] = np.einsum('ijk,ij,ki->jk', E, u, w.T).reshape(m*p)
                
                # dW[k,i,r] = sum_j E[i,j,k] * U[i,j,r] * V[j,k,r]
                # dW[:, :, r] (shape p, n)
                # dW[k,i,r] = sum_j E[i,j,k] u[i,j] v[j,k]
                # = einsum('ijk,ij,jk->ki', E, u, v)
                dW[:, r] = np.einsum('ijk,ij,jk->ki', E, u, v).reshape(p*n)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Optional: decay learning rate if error is not decreasing?
            # For now, keep it simple.
            
        return best_U, best_V, best_W, best_err

    for r in candidate_ranks:
        # Try a few restarts for each rank
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            # Set random seed for reproducibility within run, but different per attempt?
            # np.random.seed(r * 100 + attempt)
            
            U, V, W, err = optimize_rank(r, time_limit=15.0)
            
            # Verify strictly
            is_valid, final_err = verify_decomposition(U, V, W, tol=1e-4)
            
            if is_valid:
                # Found a valid decomposition at rank r
                # Convert to integer/half-integer if possible? 
                # The prompt says "integer/half-integer entries".
                # Gradient descent yields floats.
                # We need to round to nearest 0.5 or 1.0 and re-verify.
                # If rounding breaks it, we can't use it.
                # However, often these decompositions have integer structure.
                # Let's try rounding to nearest 0.5.
                
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                is_valid_rounded, err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-4)
                
                if is_valid_rounded:
                    # Return the rounded version
                    return {
                        "U": U_rounded.tolist(), 
                        "V": V_rounded.tolist(), 
                        "W": W_rounded.tolist()
                    }
                else:
                    # If rounding failed, but original was valid, we might try to keep floats?
                    # "integer/half-integer entries" is a constraint.
                    # If we can't satisfy it, this rank is invalid for the strict contract.
                    # But the evaluator might accept floats if they are exact?
                    # "integer/half-integer entries that exactly decompose"
                    # This implies the output *must* be int/half-int.
                    # So if rounding fails, we discard this solution.
                    pass
                    
        # If we didn't find a valid rounded solution for this rank, try next higher rank

    # Fallback: Schoolbook Rank 30
    # This is always valid and uses integers.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
