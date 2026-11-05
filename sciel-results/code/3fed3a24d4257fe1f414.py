import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def make_tensor(n, m, p):
        """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = make_tensor(n, m, p)
    
    def verify_fastest(U, V, W):
        """Fast verification using einsum."""
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-9)
    
    def schoolbook(n, m, p):
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def try_ranked_decomposition(target_rank, max_restarts=20, max_als_iters=50):
        """
        Try to find a decomposition of exactly target_rank using random initialization
        and ALS optimization, then check if it can be rounded to integer/half-integer.
        """
        if time.time() > deadline - 20.0:
            return None
        
        best_solution = None
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Random initialization with small values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # ALS optimization
            for it in range(max_als_iters):
                if time.time() > deadline - 10.0:
                    break
                
                # Update U
                for i in range(n*m):
                    A = np.column_stack([np.outer(V[:, r], W[:, r]) for r in range(target_rank)])
                    b = T[i, :, :].flatten()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i, :] = sol
                    except:
                        pass
                
                # Update V
                for j in range(m*p):
                    A = np.column_stack([np.outer(U[:, r], W[:, r]) for r in range(target_rank)])
                    b = T[:, j, :].flatten()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j, :] = sol
                    except:
                        pass
                
                # Update W
                for k in range(p*n):
                    A = np.column_stack([np.outer(U[:, r], V[:, r]) for r in range(target_rank)])
                    b = T[:, :, k].flatten()
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k, :] = sol
                    except:
                        pass
                
                # Check convergence
                if it > 5:
                    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
                    err = np.max(np.abs(computed - T))
                    if err < best_err:
                        best_err = err
                        best_solution = (U.copy(), V.copy(), W.copy())
                    if err < 1e-10:
                        break
            
            # Check if we can round
            if best_solution is not None and best_err < 1e-8:
                U_r, V_r, W_r = best_solution
                # Try rounding to integer or half-integer
                U_rounded = np.round(U_r * 2) / 2.0
                V_rounded = np.round(V_r * 2) / 2.0
                W_rounded = np.round(W_r * 2) / 2.0
                
                if verify_fastest(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Start with schoolbook
    U, V, W = schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various target ranks from low to high
    # Known lower bounds: for M_{3,3,8}, rank >= 23 (from M_{3,3,3} lower bound)
    # Try ranks from 23 up to 71
    for target_rank in range(23, 72):
        if time.time() > deadline - 30.0:
            break
        
        result = try_ranked_decomposition(target_rank, max_restarts=5, max_als_iters=30)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            if verify_fastest(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break  # Found a valid solution, no need to try higher ranks
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    # Remove zero columns to minimize rank
    def remove_zero_cols(U_l, V_l, W_l):
        rank = len(U_l[0]) if U_l else 0
        keep = []
        for r in range(rank):
            u_col = [row[r] for row in U_l]
            v_col = [row[r] for row in V_l]
            w_col = [row[r] for row in W_l]
            if any(abs(x) > 1e-9 for x in u_col) or \
               any(abs(x) > 1e-9 for x in v_col) or \
               any(abs(x) > 1e-9 for x in w_col):
                keep.append(r)
        
        U_new = [[row[r] for r in keep] for row in U_l]
        V_new = [[row[r] for r in keep] for row in V_l]
        W_new = [[row[r] for r in keep] for row in W_l]
        return U_new, V_new, W_new
    
    U_final, V_final, W_final = remove_zero_cols(U_list, V_list, W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
