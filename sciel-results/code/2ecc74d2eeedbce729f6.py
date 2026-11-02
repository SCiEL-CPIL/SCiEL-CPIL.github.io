import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # Safety margin

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

    def verify_solution(sol, n_trials=100):
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
    if not verify_solution(sol11, n_trials=10):
        return sol11

    def try_rank10():
        if time_remaining() < 30:
            return None
            
        rank = 10
        best_error = float('inf')
        best_params = None
        
        n_vars = n*m*rank + m*p*rank + p*n*rank
        
        # Target tensor is all ones
        T_target = np.ones((n, m, p))

        def objective(params, penalty_weight=10.0):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W, optimize=True)
            
            tensor_error = np.sum((T_computed - T_target)**2)
            
            # Penalty for deviation from half-integers
            # We want 2*x to be integer. sin^2(pi * 2x) is 0 when 2x is integer.
            pen_U = np.sum(np.sin(np.pi * 2 * U)**2)
            pen_V = np.sum(np.sin(np.pi * 2 * V)**2)
            pen_W = np.sum(np.sin(np.pi * 2 * W)**2)
            
            return tensor_error + penalty_weight * (pen_U + pen_V + pen_W)

        def params_from_sol(sol):
            U = np.array(sol["U"], dtype=float).reshape(n, m, -1)
            V = np.array(sol["V"], dtype=float).reshape(m, p, -1)
            W = np.array(sol["W"], dtype=float).reshape(p, n, -1)
            r = U.shape[2]
            U10 = U[:, :, :rank]
            V10 = V[:, :, :rank]
            W10 = W[:, :, :rank]
            return np.concatenate([U10.ravel(), V10.ravel(), W10.ravel()])

        # Strategy 1: Start from rank-11 truncated
        params0 = params_from_sol(sol11)
        res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
        
        # Strategy 2: Structured initialization - perturb the last 2 terms of rank-11
        # The last 2 terms handle the k=2 (third column) cases
        # Try to merge them into the existing 8 terms
        U11 = np.array(sol11["U"], dtype=float).reshape(n, m, 11)
        V11 = np.array(sol11["V"], dtype=float).reshape(m, p, 11)
        W11 = np.array(sol11["W"], dtype=float).reshape(p, n, 11)
        
        # Create a structured init: keep first 8 terms, blend terms 9-10 into existing
        for attempt in range(5):
            if time_remaining() < 15:
                break
            U_struct = U11[:, :, :rank].copy()
            V_struct = V11[:, :, :rank].copy()
            W_struct = W11[:, :, :rank].copy()
            
            # Add small perturbations to encourage merging
            noise_scale = 0.1 * (attempt + 1)
            U_struct += np.random.randn(n, m, rank) * noise_scale
            V_struct += np.random.randn(m, p, rank) * noise_scale
            W_struct += np.random.randn(p, n, rank) * noise_scale
            
            params0 = np.concatenate([U_struct.ravel(), V_struct.ravel(), W_struct.ravel()])
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
        
        # Strategy 3: Random restarts
        n_restarts = 300
        for i in range(n_restarts):
            if time_remaining() < 10:
                break
                
            params0 = np.random.randn(n_vars) * 1.0
            
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-10:
                break
        
        if best_error < 1e-6 and best_params is not None:
            # Polish pass with tighter tolerance
            if time_remaining() > 5:
                res_polish = minimize(objective, best_params, method='L-BFGS-B', 
                                       options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
                if res_polish.fun < best_error:
                    best_error = res_polish.fun
                    best_params = res_polish.x
            
            U_flat = best_params[:n*m*rank]
            V_flat = best_params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = best_params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            # Snap to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Re-verify with snapped values
            U_list = [[0]*rank for _ in range(n*m)]
            V_list = [[0]*rank for _ in range(m*p)]
            W_list = [[0]*rank for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        U_list[i*m+j][r] = U[i,j,r]
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        V_list[j*p+k][r] = V[j,k,r]
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        W_list[k*n+i][r] = W[k,i,r]
                        
            sol10 = {"U": U_list, "V": V_list, "W": W_list}
            
            if verify_solution(sol10, n_trials=50):
                return sol10
                
        return None

    sol10 = try_rank10()
    if sol10 is not None:
        return sol10
    else:
        return sol11
