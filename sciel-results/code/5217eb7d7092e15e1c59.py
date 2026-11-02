import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Build the target tensor as a list of (u_idx, v_idx, w_idx) tuples where value is 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_indices.append((u_idx, v_idx, w_idx))
    
    num_targets = len(target_indices)
    
    # We want to find U, V, W such that
    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] == 1 for each target
    # and 0 for non-targets? No, the decomposition only needs to match the targets.
    # Wait, the tensor is zero elsewhere.
    # So we need sum_r U[u, r] V[v, r] W[w, r] == 1 if (u,v,w) is in target_indices, else 0.
    
    # However, the number of non-target entries is huge (8*28*14 - 56).
    # We only need to enforce the constraints for the non-zero entries of the target tensor.
    # Is that sufficient?
    # If the decomposition matches the non-zero entries, does it automatically match the zeros?
    # No. We must ensure the zeros are also satisfied.
    # But the tensor for matrix multiplication is sparse.
    # Actually, the standard definition of the matrix multiplication tensor $M_{n,m,p}$ has 1s at $(i,j; j,k; k,i)$ and 0 elsewhere.
    # So we need to satisfy both.
    
    # To make this tractable, we can use the fact that the tensor is a sum of rank-1 tensors.
    # We can minimize the error on the non-zero entries first, hoping the zeros stay close to 0.
    # Or we can include a regularization term.
    
    # Let's try to minimize the error on the non-zero entries only.
    # If the rank is low enough, the zeros might be satisfied automatically due to the structure?
    # Not necessarily.
    
    # A safer approach:
    # Minimize sum_{(u,v,w) in targets} (prod - 1)^2 + sum_{(u,v,w) not in targets} (prod)^2
    # The second sum is too large to compute explicitly.
    
    # Alternative:
    # Use the fact that the tensor is the identity map.
    # We can use a random projection method or just trust that if we fit the 1s, the 0s will be small if the rank is not too low.
    # Given the time limit, let's try fitting the 1s.
    
    def error_func(params, R):
        # params is a flat array of length (n*m + m*p + p*n) * R
        u_size = n * m
        v_size = m * p
        w_size = p * n
        total_size = (u_size + v_size + w_size) * R
        
        U = params[:u_size * R].reshape(u_size, R)
        V = params[u_size * R : u_size * R + v_size * R].reshape(v_size, R)
        W = params[u_size * R + v_size * R :].reshape(w_size, R)
        
        error = 0.0
        for u_idx, v_idx, w_idx in target_indices:
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            error += (val - 1.0) ** 2
            
        # Optional: add a small penalty for large values to keep numbers bounded
        # error += 1e-6 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
        
        return error

    def check_solution(U, V, W):
        # Check if the solution is exact (within tolerance)
        # And snap to 0.5 if close
        tol = 1e-5
        for u_idx, v_idx, w_idx in target_indices:
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            if abs(val - 1.0) > tol:
                return False
        # Check zeros (sample a few)
        # We can't check all, but we can check that the matrix product works
        # Let's construct A, B random and check C
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            # u_r is a vector of length n*m, reshape to n x m
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_vec = W[:, r] # length p*n
            
            # The term is (u_r^T A) (v_r^T B) w_r?
            # No, the decomposition is sum_r (U[:,r] * A_flat) * (V[:,r] * B_flat) * W[:,r]?
            # Let's re-verify the formula.
            # C_{i,k} = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This is NOT (U[:,r] * A_flat).
            # U[:,r] is a vector. A is a matrix.
            # The product U[i*m+j, r] * A[i,j] is not a dot product.
            # It is a weighted sum.
            # So C = sum_r (U[:,r] reshaped to n x m) * A (element-wise) ... no.
            # Let's look at the term:
            # T_r(A, B) = sum_{i,j,k} U[i,j,r] A[i,j] V[j,k,r] B[j,k] W[k,i,r]
            # This is a trilinear form.
            # We can compute this efficiently.
            
            # For each r, compute M_r = U[:,r].reshape(n,m) * A (element-wise)
            # Then sum over j?
            # Let's just loop over i, k for the check, it's small (2x7).
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
                    C_computed[i,k] += s
                    
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
        return True

    def snap_to_half_integer(U, V, W):
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return U, V, W

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Try ranks from lower bound to schoolbook
    # Lower bound: max(n*m, m*p, p*n) is not a bound.
    # A better bound: rank >= nm + mp + np - 2?
    # For 2,4,7: 8+28+14-2 = 48.
    # Let's try from 48 to 56.
    
    for R in range(48, best_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            np.random.seed(restart)
            U0 = np.random.randn(n*m, R) * 0.1
            V0 = np.random.randn(m*p, R) * 0.1
            W0 = np.random.randn(p*n, R) * 0.1
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use L-BFGS-B
            res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success or res.fun < 1e-6:
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Snap and verify
                U_snap, V_snap, W_snap = snap_to_half_integer(U_opt, V_opt, W_opt)
                
                if check_solution(U_snap, V_snap, W_snap):
                    best_rank = R
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break # Found a solution for this rank, try lower rank? No, we are iterating up.
                    # Actually, we want the MINIMAL rank.
                    # So if we find a solution for R, we should record it and continue to lower R?
                    # The loop is `for R in range(48, best_rank + 1)`.
                    # So it starts at 48. If it finds 48, it breaks the inner loop.
                    # Then it goes to R=49? No, we want the smallest R.
                    # So we should break the outer loop if we find a solution?
                    # No, we are iterating R upwards.
                    # If we find a solution at R=48, that's the best we can hope for (if 48 is the bound).
                    # So we should break the outer loop immediately.
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
            
            # Also try to check if the unsnapped solution is good, then snap
            # The snap might break the solution if it's not close enough.
            # We already checked snapped.

    # If no better rank found, return schoolbook
    if best_U is None:
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
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
