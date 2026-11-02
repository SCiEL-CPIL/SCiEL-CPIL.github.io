import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def compute_error(U_np, V_np, W_np):
        """
        Compute the sum of squared errors for all (i,j,k).
        """
        rank = U_np.shape[1]
        S = np.zeros((n, m, p))
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            S += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        error = np.sum((S - 1.0) ** 2)
        return error

    def verify_fast(U_np, V_np, W_np, tol=1e-9):
        """Verify if the decomposition is exact."""
        S = np.zeros((n, m, p))
        rank = U_np.shape[1]
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            S += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        return np.allclose(S, 1.0, atol=tol)

    def quantize(arr):
        """Quantize to nearest half-integer."""
        return np.round(arr * 2) / 2

    def polish_discrete(U_np, V_np, W_np, max_steps=100):
        """
        Local search in the discrete space of half-integers to reduce error.
        """
        U_np = quantize(U_np)
        V_np = quantize(V_np)
        W_np = quantize(W_np)
        
        current_error = compute_error(U_np, V_np, W_np)
        if current_error < 1e-9:
            return U_np, V_np, W_np, True

        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
        best_err = current_error
        
        # Define all indices
        U_indices = [(i, j) for i in range(n*m) for j in range(U_np.shape[1])]
        V_indices = [(i, j) for i in range(m*p) for j in range(V_np.shape[1])]
        W_indices = [(i, j) for i in range(p*n) for j in range(W_np.shape[1])]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try random subset of variables to avoid full scan cost
            num_trials = 50
            for _ in range(num_trials):
                # Pick a random variable from U, V, or W
                choice = random.randint(0, 2)
                if choice == 0:
                    idx, jdx = random.choice(U_indices)
                    old_val = U_np[idx, jdx]
                    # Try neighbors: old_val +/- 0.5
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        U_np[idx, jdx] = new_val
                        err = compute_error(U_np, V_np, W_np)
                        if err < best_err - 1e-12:
                            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
                            best_err = err
                            improved = True
                            break
                        U_np[idx, jdx] = old_val
                elif choice == 1:
                    idx, jdx = random.choice(V_indices)
                    old_val = V_np[idx, jdx]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        V_np[idx, jdx] = new_val
                        err = compute_error(U_np, V_np, W_np)
                        if err < best_err - 1e-12:
                            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
                            best_err = err
                            improved = True
                            break
                        V_np[idx, jdx] = old_val
                else:
                    idx, jdx = random.choice(W_indices)
                    old_val = W_np[idx, jdx]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        W_np[idx, jdx] = new_val
                        err = compute_error(U_np, V_np, W_np)
                        if err < best_err - 1e-12:
                            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
                            best_err = err
                            improved = True
                            break
                        W_np[idx, jdx] = old_val
            
            if not improved:
                break
            if best_err < 1e-9:
                break

        return best_U, best_V, best_W, best_err < 1e-9

    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]

    # Schoolbook solution as fallback
    U_sch = np.zeros((n * m, schoolbook_rank))
    V_sch = np.zeros((m * p, schoolbook_rank))
    W_sch = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = schoolbook_rank

    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Construct a rank-49 decomposition by fixing the matrix column index k
    # and applying the rank-16 Kronecker product structure of the 4x4 matrix-vector
    # multiplication to each column independently.
    
    # First, try to construct a rank-49 solution using Kronecker product structure
    # For each k in [0, p), we have a 4x4 matrix-vector multiplication (A @ e_k)
    # where e_k is the k-th standard basis vector.
    # The rank of 4x4 matrix-vector multiplication is at most 16 (schoolbook).
    # So total rank would be 7 * 16 = 112, same as schoolbook.
    
    # But we can potentially do better by sharing structure across k values.
    
    # Let's try a direct construction with rank 49:
    # We'll use a randomized approach with the constraint that the solution
    # has a specific Kronecker product structure.
    
    # Actually, let's first try to find any solution with rank < 112 using
    # the existing optimization approach, but with a better initialization.
    
    # Search ranks from 40 to 111
    # We prioritize lower ranks.
    for target_rank in range(40, 112):
        if time.time() > deadline - 20:
            break
            
        if target_rank >= best_rank:
            continue
            
        # Number of restarts
        num_restarts = 5
        if target_rank < 50:
            num_restarts = 8
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization with structure
            # Use the idea that for each k, the sub-problem is 4x4 matrix-vector
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Apply sparsity mask
            U_r[np.random.rand(n*m, target_rank) < 0.5] = 0
            V_r[np.random.rand(m*p, target_rank) < 0.5] = 0
            W_r[np.random.rand(p*n, target_rank) < 0.5] = 0
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank:]
                return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
            
            def objective(params):
                U, V, W = unflatten_params(params)
                return compute_error(U, V, W)
            
            initial_params = flatten_params(U_r, V_r, W_r)
            
            try:
                res = minimize(objective, initial_params, method='L-BFGS-B', 
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
                
                U_opt, V_opt, W_opt = unflatten_params(res.x)
                
                # Quantize
                U_q = quantize(U_opt)
                V_q = quantize(V_opt)
                W_q = quantize(W_opt)
                
                # Check if exact
                if verify_fast(U_q, V_q, W_q):
                    best_U, best_V, best_W = U_q, V_q, W_q
                    best_rank = target_rank
                    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
                
                # If not exact, try discrete polish
                U_pol, V_pol, W_pol, success = polish_discrete(U_opt, V_opt, W_opt)
                if success:
                    best_U, best_V, best_W = U_pol, V_pol, W_pol
                    best_rank = target_rank
                    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
                    
            except Exception as e:
                pass

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
