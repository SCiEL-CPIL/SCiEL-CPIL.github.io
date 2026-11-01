import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = time.time() + 145.0
    
    # Schoolbook fallback
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-6)

    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-6)

    def to_half_int_list(mat):
        """Convert matrix to list, ensuring half-integer values"""
        doubled = np.round(mat * 2)
        return doubled.astype(int).tolist()

    def objective(params, n, m, p, r):
        """
        Objective function for minimizing the error in tensor decomposition.
        params: flattened array of U, V, W matrices.
        Returns sum of squared errors.
        """
        # Unpack
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        
        # Compute T[i,k,j] = sum_c U[i,k,c] V[k,j,c] W[j,i,c]
        T = np.einsum('ikc,kjc,jic->ijk', U, V, W)
        
        # Target is 1 for all i,k,j
        error = np.sum((T - 1.0)**2)
        return error

    def try_optimize(rank, max_iter=500):
        """
        Attempts to find a rank decomposition using numerical optimization.
        """
        if time.time() > deadline:
            return None
            
        # Initialize with schoolbook columns (first 'rank' columns of schoolbook)
        # This provides a valid starting point that can be pruned/optimized
        U_init = np.zeros((n, m, rank))
        V_init = np.zeros((m, p, rank))
        W_init = np.zeros((p, n, rank))
        
        # Fill with schoolbook terms for the first 'rank' indices
        # Schoolbook indices: i, k, j -> index = i*(m*p) + k*p + j
        # We want to pick 'rank' distinct (i,k,j) tuples
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    if idx < rank:
                        U_init[i, k, idx] = 1.0
                        V_init[k, j, idx] = 1.0
                        W_init[j, i, idx] = 1.0
                        idx += 1
                    else:
                        break
            if idx >= rank: break
        if idx >= rank: break

        params_init = np.concatenate([
            U_init.flatten(), 
            V_init.flatten(), 
            W_init.flatten()
        ])
        
        try:
            res = minimize(
                objective, 
                params_init, 
                args=(n, m, p, rank), 
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or (res.fun < 1e-5):
                # Reconstruct matrices
                size_U = n * m * rank
                size_V = m * p * rank
                size_W = p * n * rank
                
                U_opt = res.x[:size_U].reshape(n, m, rank)
                V_opt = res.x[size_U:size_U + size_V].reshape(m, p, rank)
                W_opt = res.x[size_U + size_V:].reshape(p, n, rank)
                
                # Convert to (rows, rank) format for verification
                U_mat = U_opt.transpose(1, 0, 2).reshape(n*m, rank)
                V_mat = V_opt.transpose(1, 0, 2).reshape(m*p, rank)
                W_mat = W_opt.transpose(0, 1, 2).reshape(p*n, rank)
                
                if verify(U_mat, V_mat, W_mat):
                    if is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                        return {
                            "U": to_half_int_list(U_mat),
                            "V": to_half_int_list(V_mat),
                            "W": to_half_int_list(W_mat)
                        }
                    else:
                        # Try rounding
                        U_round = np.round(U_mat * 2) / 2
                        V_round = np.round(V_mat * 2) / 2
                        W_round = np.round(W_mat * 2) / 2
                        if verify(U_round, V_round, W_round):
                            return {
                                "U": to_half_int_list(U_round),
                                "V": to_half_int_list(V_round),
                                "W": to_half_int_list(W_round)
                            }
        except Exception:
            pass
            
        return None

    # Search for lower ranks
    # Theoretical lower bound for 2x2x7 is likely around 20-24.
    # Schoolbook is 28.
    # We try ranks from 27 down to 15.
    
    # Note: L-BFGS-B on high-dimensional non-convex problems can be slow.
    # We limit iterations per rank to keep within time budget.
    
    current_time = time.time()
    
    # Strategy: Try ranks in decreasing order.
    # If we find a valid one, return it.
    
    # We can also try random initializations if schoolbook init fails to converge to low error
    # but for now, let's stick to schoolbook init as it's structurally close.
    
    for r in range(best_rank - 1, 14, -1):
        if time.time() > deadline - 5:
            break
            
        # Run optimization
        result = try_optimize(r, max_iter=100)
        
        if result is not None:
            best = result
            best_rank = r
            # Since we are searching downwards, the first valid one is the best found in this loop
            # However, we might find a lower rank later? No, we go from high to low.
            # So the first one found is the lowest rank we've found so far in this direction?
            # Wait, we are iterating from 27 down to 15.
            # If 27 works, we record it. Then we try 26. If 26 works, it's better.
            # So we continue searching.
            
    # Return the best found
    return best
# EVOLVE-BLOCK-END
