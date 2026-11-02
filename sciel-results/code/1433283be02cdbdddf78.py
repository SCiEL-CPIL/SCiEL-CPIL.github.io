import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145

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
        if time_remaining() < 40:
            return None
            
        rank = 10
        best_error = float('inf')
        best_params = None
        
        # Number of variables: n*m*rank + m*p*rank + p*n*rank
        # 4*10 + 6*10 + 6*10 = 160 variables
        
        def objective(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            # Calculate tensor T_computed[i,j,k]
            # T_computed[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            
            # Target tensor T_target[i,j,k] = 1 for all i,j,k
            T_target = np.ones((n, m, p))
            
            error = np.sum((T_computed - T_target)**2)
            return error

        def params_from_sol(sol):
            U = np.array(sol["U"], dtype=float).reshape(n, m, -1)
            V = np.array(sol["V"], dtype=float).reshape(m, p, -1)
            W = np.array(sol["W"], dtype=float).reshape(p, n, -1)
            r = U.shape[2]
            # Truncate to rank 10
            U10 = U[:, :, :rank]
            V10 = V[:, :, :rank]
            W10 = W[:, :, :rank]
            return np.concatenate([U10.ravel(), V10.ravel(), W10.ravel()])

        # Strategy 1: Start from rank-11 truncated to rank-10
        params0 = params_from_sol(sol11)
        res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-12})
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
        
        # Strategy 2: Multiple random restarts with better scaling
        n_restarts = 12
        for i in range(n_restarts):
            if time_remaining() < 20:
                break
                
            # Random initialization with appropriate scale
            params0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.5
            
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 1200, 'ftol': 1e-15, 'gtol': 1e-11})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-10:
                break
        
        if best_error < 1e-8:
            U_flat = best_params[:n*m*rank]
            V_flat = best_params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = best_params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            # Convert to list of lists for return
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
            
            # Verify
            if verify_solution(sol10, n_trials=50):
                return sol10
                
        return None

    sol10 = try_rank10()
    if sol10 is not None:
        return sol10
    else:
        return sol11

# EVOLVE-BLOCK-END
