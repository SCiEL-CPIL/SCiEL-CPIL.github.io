import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    # Schoolbook
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    # Strategy: Decompose 3x4x8 into two 3x4x4 multiplications.
    # If we find a rank R_sub decomposition for 3x4x4,
    # the total rank is 2 * R_sub.
    # Schoolbook for 3x4x4 is 48. 2*48 = 96.
    # We want R_sub < 48, e.g., 47 or 46.
    
    n_sub, m_sub, p_sub = 3, 4, 4
    
    def solve_sub(n_s, m_s, p_s, R_target, time_limit):
        """
        Attempts to find a rank R_target decomposition for n_s x m_s x p_s.
        Returns (U, V, W) if found, else None.
        """
        def error_and_grad(x):
            U_flat, V_flat, W_flat = x[:n_s*m_s*R_target], x[n_s*m_s*R_target:n_s*m_s*R_target+m_s*p_s*R_target], x[n_s*m_s*R_target+m_s*p_s*R_target:]
            U_c = U_flat.reshape(n_s*m_s, R_target)
            V_c = V_flat.reshape(m_s*p_s, R_target)
            W_c = W_flat.reshape(p_s*n_s, R_target)
            
            err = 0.0
            grad_U = np.zeros_like(U_c)
            grad_V = np.zeros_like(V_c)
            grad_W = np.zeros_like(W_c)
            
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        prod = V_c[j*p_s+k, :] * W_c[k*n_s+i, :]
                        val = np.dot(U_c[i*m_s+j, :], prod)
                        diff = val - 1.0
                        err += diff * diff
                        
                        factor = 2.0 * diff
                        grad_U[i*m_s+j, :] += factor * prod
                        grad_V[j*p_s+k, :] += factor * U_c[i*m_s+j, :] * W_c[k*n_s+i, :]
                        grad_W[k*n_s+i, :] += factor * U_c[i*m_s+j, :] * V_c[j*p_s+k, :]
            
            return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        np.random.seed(42)
        best_err = np.inf
        best_x = None
        
        # Try a few seeds
        for seed in [42, 123, 999, 777]:
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(seed)
            # Initialize with small random values
            U = np.random.randn(n_s*m_s, R_target) * 0.5
            V = np.random.randn(m_s*p_s, R_target) * 0.5
            W = np.random.randn(p_s*n_s, R_target) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(error_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-5})
                if res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x
                    if best_err < 1e-6:
                        break
            except Exception:
                pass

        if best_x is not None and best_err < 1e-5:
            U_sol = best_x[:n_s*m_s*R_target].reshape(n_s*m_s, R_target)
            V_sol = best_x[n_s*m_s*R_target:n_s*m_s*R_target+m_s*p_s*R_target].reshape(m_s*p_s, R_target)
            W_sol = best_x[n_s*m_s*R_target+m_s*p_s*R_target:].reshape(p_s*n_s, R_target)
            
            # Round to half-integers
            U_round = np.round(U_sol * 2) / 2.0
            V_round = np.round(V_sol * 2) / 2.0
            W_round = np.round(W_sol * 2) / 2.0
            
            # Verify
            err_round = 0.0
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        val = np.dot(U_round[i*m_s+j, :], V_round[j*p_s+k, :] * W_round[k*n_s+i, :])
                        err_round += (val - 1.0)**2
            
            if err_round < 1e-4:
                return U_round, V_round, W_round
            
        return None

    # Try to find rank 47 for 3x4x4
    if time.time() - start_time < timeout - 10:
        sub_sol = solve_sub(n_sub, m_sub, p_sub, 47, timeout - 20)
        if sub_sol:
            U_sub, V_sub, W_sub = sub_sol
            R_sub = 47
            R_total = 2 * R_sub
            
            # Construct 3x4x8 solution
            U_total = np.zeros((n*m, R_total))
            V_total = np.zeros((m*p, R_total))
            W_total = np.zeros((p*n, R_total))
            
            # U is shared
            U_total[:, :R_sub] = U_sub
            U_total[:, R_sub:] = U_sub
            
            # V: First R_sub columns correspond to B_left (cols 0-3 of B)
            # V_sub is 4x4 x R_sub. Indices j in 0..3, k in 0..3.
            # In V_total, j in 0..3, k in 0..3.
            for j in range(m_sub):
                for k in range(p_sub):
                    V_total[j*p + k, :R_sub] = V_sub[j*p_sub + k, :]
            
            # Second R_sub columns correspond to B_right (cols 4-7 of B)
            # V_sub indices j in 0..3, k in 0..3 map to k in 4..7 in V_total.
            for j in range(m_sub):
                for k in range(p_sub):
                    V_total[j*p + (k + p_sub), R_sub:] = V_sub[j*p_sub + k, :]
            
            # W: First R_sub columns correspond to C_left (cols 0-3 of C, i.e., k in 0..3 for W index k*n+i)
            # W_sub is 4x3 x R_sub. Indices k in 0..3, i in 0..2.
            # In W_total, k in 0..3.
            for k in range(p_sub):
                for i in range(n_sub):
                    W_total[k*n + i, :R_sub] = W_sub[k*n_sub + i, :]
            
            # Second R_sub columns correspond to C_right (cols 4-7 of C, i.e., k in 4..7 for W index k*n+i)
            # W_sub indices k in 0..3, i in 0..2 map to k in 4..7 in W_total.
            for k in range(p_sub):
                for i in range(n_sub):
                    W_total[(k + p_sub)*n + i, R_sub:] = W_sub[k*n_sub + i, :]
            
            # Verify full solution
            err_full = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_total[i*m+j, :], V_total[j*p+k, :] * W_total[k*n+i, :])
                        err_full += (val - 1.0)**2
            
            if err_full < 1e-4:
                best_U = U_total
                best_V = V_total
                best_W = W_total
                best_rank = R_total

    if best_rank < rank_seed:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
