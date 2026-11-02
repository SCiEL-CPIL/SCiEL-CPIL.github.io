import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # The known rank-11 decomposition
    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def set_u(i, j, r, val): U[i * m + j][r] = val
        def set_v(j, k, r, val): V[j * p + k][r] = val
        def set_w(k, i, r, val): W[k * n + i][r] = val

        # Term 1: (A00 + A11) * (B00 + B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # Term 2: (A10 + A11) * B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # Term 3: A00 * (B01 - B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # Term 4: A11 * (B10 - B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # Term 5: (A00 + A01) * B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # Term 6: (A10 - A00) * (B00 + B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # Term 7: (A01 - A11) * (B10 + B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # Term 8: A00 * B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # Term 9: A01 * B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # Term 10: A10 * B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # Term 11: A11 * B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify_solution(sol, n_trials=20):
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            M = np.einsum('ijr,ij->r', U3, A)
            N = np.einsum('jkr,jk->r', V3, B)
            MN = M * N
            C_computed = np.einsum('kir,r->ik', W3, MN)
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    sol11 = make_rank11()
    if not verify_solution(sol11, n_trials=5):
        return sol11

    def try_rank10():
        if time_remaining() < 50:
            return None
            
        rank = 10
        
        # Get the rank-11 solution as arrays
        U11 = np.array(sol11["U"], dtype=float).reshape(n, m, 11)
        V11 = np.array(sol11["V"], dtype=float).reshape(m, p, 11)
        W11 = np.array(sol11["W"], dtype=float).reshape(p, n, 11)
        
        # Strategy: Try to merge pairs of terms from the rank-11 solution
        # For each pair (r1, r2), try to find a single term that replaces both
        # while maintaining the tensor identity
        
        best_error = float('inf')
        best_params = None
        
        # For each pair of terms to merge
        for r1, r2 in combinations(range(11), 2):
            if time_remaining() < 20:
                break
                
            # Keep terms 0..r1-1, r2+1..10 (10 terms total after merging r1,r2)
            keep = [r for r in range(11) if r not in [r1, r2]]
            
            # The merged term will be at position len(keep) in the new rank-10 solution
            # We need to find U_merged, V_merged, W_merged such that:
            # sum_{r in keep} U_r V_r W_r + U_merged V_merged W_merged = T_target
            
            # This is equivalent to: U_merged V_merged W_merged = T_target - sum_{r in keep} U_r V_r W_r
            # Let T_residual = T_target - sum_{r in keep} U_r V_r W_r
            # We need to find a rank-1 decomposition of T_residual
            
            # T_residual[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # For the kept terms:
            T_residual = np.zeros((n, m, p))
            for r in keep:
                T_residual += U11[:, :, r] * V11[:, :, r] * W11[:, :, r]
            
            # Target is all ones
            T_target = np.ones((n, m, p))
            T_to_decompose = T_target - T_residual
            
            # Now try to find rank-1 decomposition of T_to_decompose
            # T_to_decompose[i,j,k] = U_new[i,j] * V_new[j,k] * W_new[k,i]
            
            # This is a system of 2*2*3 = 12 equations with 4+6+6 = 16 unknowns
            # But we can use the fact that it must be rank-1 in the tensor sense
            
            # Try to solve: for each (i,j,k), T_to_decompose[i,j,k] = U[i,j]*V[j,k]*W[k,i]
            # We can fix W[k,i] = 1 for some (k,i) to remove scale ambiguity
            # Then solve for U and V
            
            # Better approach: use the structure. For rank-1 tensor:
            # T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
            # Taking logs: log|T[i,j,k]| = log|a[i,j]| + log|b[j,k]| + log|c[k,i]|
            # But T can be negative, so this is tricky.
            
            # Alternative: use the fact that for any fixed k, T[:, :, k] must be rank-1
            # T[:, :, k] = a[:, :] * b[:, k] * c[k, :]
            # So T[:, :, k] = (a[:, :] * c[k, :]) * b[:, k]
            # This means T[:, :, k] is an outer product of two vectors
            
            # Check if T_to_decompose[:, :, k] is rank-1 for all k
            # If yes, we can extract the factors
            
            valid = True
            for k in range(p):
                slice_k = T_to_decompose[:, :, k]
                # Check rank-1: all 2x2 minors should be zero
                # slice_k[0,0]*slice_k[1,1] - slice_k[0,1]*slice_k[1,0] == 0
                if abs(slice_k[0,0]*slice_k[1,1] - slice_k[0,1]*slice_k[1,0]) > 1e-10:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Now extract the rank-1 factors
            # For k=0: T[:, :, 0] = a[:, :] * b[:, 0] * c[0, :]
            # Let's set c[0, i] = 1 for i=0 (fix scale)
            # Then a[i, j] * b[j, 0] = T[i, j, 0]
            
            # Try different scale fixes
            found = False
            for k_fix in range(p):
                for i_fix in range(n):
                    # Fix c[k_fix, i_fix] = 1
                    # Then for all (i, j, k): T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
                    # With c[k_fix, i_fix] = 1
                    
                    # From T[i, j, k_fix] = a[i,j] * b[j, k_fix] * c[k_fix, i]
                    # And T[i_fix, j, k] = a[i_fix,j] * b[j,k] * c[k, i_fix]
                    
                    # This is still complex. Let's use a direct approach:
                    # Assume a[0,0] = 1 (fix scale)
                    # Then b[j,0] = T[0,j,0] / a[0,j] ... circular
                    
                    # Better: use the structure of the known solution
                    # The terms in the rank-11 solution have specific patterns
                    # Let's try to see if the residual has a similar pattern
                    
                    # For now, skip this complex extraction and use a numerical approach
                    break
                break
            
            # Use numerical optimization to find rank-1 decomposition of T_to_decompose
            # Variables: U_new (4), V_new (6), W_new (6) = 16 variables
            # Equations: T_to_decompose[i,j,k] = U_new[i,j]*V_new[j,k]*W_new[k,i] for 12 values
            
            if time_remaining() < 15:
                break
                
            # Initialize with small random values
            np.random.seed(42)  # Fixed seed for reproducibility
            U_new = np.random.randn(n, m) * 0.5
            V_new = np.random.randn(m, p) * 0.5
            W_new = np.random.randn(p, n) * 0.5
            
            from scipy.optimize import minimize
            
            def objective(params):
                U_flat = params[:n*m]
                V_flat = params[n*m:n*m+m*p]
                W_flat = params[n*m+m*p:]
                
                U = U_flat.reshape(n, m)
                V = V_flat.reshape(m, p)
                W = W_flat.reshape(p, n)
                
                T_computed = U[:, :, None] * V[:, :, None].T * W[:, None, :]
                # T_computed[i,j,k] = U[i,j] * V[j,k] * W[k,i]
                # Need to be careful with indexing
                T_computed = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_computed[i,j,k] = U[i,j] * V[j,k] * W[k,i]
                
                error = np.sum((T_computed - T_to_decompose)**2)
                return error
            
            params0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            
            res = minimize(objective, params0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
                # Check if we found a good solution
                if res.fun < 1e-8:
                    # Extract the solution
                    U_new_sol = res.x[:n*m].reshape(n, m)
                    V_new_sol = res.x[n*m:n*m+m*p].reshape(m, p)
                    W_new_sol = res.x[n*m+m*p:].reshape(p, n)
                    
                    # Build the rank-10 solution
                    U10 = np.zeros((n, m, rank))
                    V10 = np.zeros((m, p, rank))
                    W10 = np.zeros((p, n, rank))
                    
                    # Copy kept terms
                    for new_r, old_r in enumerate(keep):
                        U10[:, :, new_r] = U11[:, :, old_r]
                        V10[:, :, new_r] = V11[:, :, old_r]
                        W10[:, :, new_r] = W11[:, :, old_r]
                    
                    # Add merged term
                    merged_idx = len(keep)
                    U10[:, :, merged_idx] = U_new_sol
                    V10[:, :, merged_idx] = V_new_sol
                    W10[:, :, merged_idx] = W_new_sol
                    
                    # Convert to list format
                    U_list = [[0]*rank for _ in range(n*m)]
                    V_list = [[0]*rank for _ in range(m*p)]
                    W_list = [[0]*rank for _ in range(p*n)]
                    
                    for i in range(n):
                        for j in range(m):
                            for r in range(rank):
                                U_list[i*m+j][r] = float(U10[i,j,r])
                    for j in range(m):
                        for k in range(p):
                            for r in range(rank):
                                V_list[j*p+k][r] = float(V10[j,k,r])
                    for k in range(p):
                        for i in range(n):
                            for r in range(rank):
                                W_list[k*n+i][r] = float(W10[k,i,r])
                    
                    sol10 = {"U": U_list, "V": V_list, "W": W_list}
                    
                    if verify_solution(sol10, n_trials=20):
                        return sol10
            
            if best_error < 1e-8:
                break
        
        return None

    sol10 = try_rank10()
    if sol10 is not None:
        return sol10
    else:
        return sol11

# EVOLVE-BLOCK-END
