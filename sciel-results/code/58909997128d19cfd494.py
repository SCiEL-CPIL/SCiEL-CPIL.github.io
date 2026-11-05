import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T of shape (n, m, p)
    T = np.ones((n, m, p))
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    def reconstruct(params, R):
        U_flat = params[0:R*size_U]
        V_flat = params[R*size_U:R*size_U + R*size_V]
        W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W]
        
        U = U_flat.reshape(R, size_U)
        V = V_flat.reshape(R, size_V)
        W = W_flat.reshape(R, size_W)
        
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_recon[i, j, k] = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
        return T_recon

    def loss(params, R):
        T_recon = reconstruct(params, R)
        error = np.sum((T - T_recon) ** 2)
        return error

    def snap_and_verify(params, R):
        U_flat = params[0:R*size_U]
        V_flat = params[R*size_U:R*size_U + R*size_V]
        W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W]
        
        # Try snapping to half-integers with small tolerance
        for tol in [0.01, 0.05, 0.1]:
            U_snap = np.round(U_flat / tol) * tol
            V_snap = np.round(V_flat / tol) * tol
            W_snap = np.round(W_flat / tol) * tol
            
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        T_recon[i, j, k] = np.sum(U_snap[u_idx] * V_snap[v_idx] * W_snap[w_idx])
                        
            if np.allclose(T_recon, T, atol=1e-9):
                U_out = [[float(U_snap[i*size_U + r]) for r in range(R)] for i in range(size_U)]
                V_out = [[float(V_snap[i*size_V + r]) for r in range(R)] for i in range(size_V)]
                W_out = [[float(W_snap[i*size_W + r]) for r in range(R)] for i in range(size_W)]
                return {"U": U_out, "V": V_out, "W": W_out}
        return None

    def gradient(params, R):
        T_recon = reconstruct(params, R)
        diff = T - T_recon
        
        U_flat = params[0:R*size_U]
        V_flat = params[R*size_U:R*size_U + R*size_V]
        W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W]
        
        U = U_flat.reshape(R, size_U)
        V = V_flat.reshape(R, size_V)
        W = W_flat.reshape(R, size_W)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    d = diff[i, j, k]
                    grad_U[:, u_idx] -= 2 * d * (V[:, v_idx] * W[:, w_idx])
                    grad_V[:, v_idx] -= 2 * d * (U[:, u_idx] * W[:, w_idx])
                    grad_W[:, w_idx] -= 2 * d * (U[:, u_idx] * V[:, v_idx])
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return grad_flat

    def als_refine(params, R, max_iter=100):
        """ALS refinement to converge to exact solution"""
        U_flat = params[0:R*size_U].copy()
        V_flat = params[R*size_U:R*size_U + R*size_V].copy()
        W_flat = params[R*size_U + R*size_V:R*size_U + R*size_V + R*size_W].copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            U = U_flat.reshape(R, size_U)
            V = V_flat.reshape(R, size_V)
            W = W_flat.reshape(R, size_W)
            
            # Solve for U: for each u_idx, U[:, u_idx] should satisfy
            # sum_r U[r,u_idx] * V[r,v_idx] * W[r,w_idx] = T[i,j,k]
            # This is a linear system in U[:, u_idx]
            new_U = U.copy()
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Build system: for each k, sum_r U[r,u_idx] * (V[r, j*p+k] * W[r, k*n+i]) = 1
                    # Let a_k = V[:, j*p+k] * W[:, k*n+i]
                    # Then sum_r U[r,u_idx] * a_k[r] = 1 for all k
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        A_mat[k, :] = V[:, v_idx] * W[:, w_idx]
                    b_vec = np.ones(p)
                    # Solve min ||A_mat @ U_col - b_vec||^2
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    new_U[:, u_idx] = sol
            
            # Solve for V
            new_V = V.copy()
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        A_mat[i, :] = U[:, u_idx] * W[:, w_idx]
                    b_vec = np.ones(n)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    new_V[:, v_idx] = sol
            
            # Solve for W
            new_W = W.copy()
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        A_mat[j, :] = U[:, u_idx] * V[:, v_idx]
                    b_vec = np.ones(m)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    new_W[:, w_idx] = sol
            
            U_flat = new_U.flatten()
            V_flat = new_V.flatten()
            W_flat = new_W.flatten()
            
            # Check convergence
            params_new = np.concatenate([U_flat, V_flat, W_flat])
            if loss(params_new, R) < 1e-10:
                break
        
        return np.concatenate([U_flat, V_flat, W_flat])

    best_result = None
    
    for R in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(8):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 100 + R)
            init = np.random.randn(R * (n*m + m*p + p*n)) * 0.5
            
            try:
                res = minimize(loss, init, args=(R,), method='BFGS', jac=gradient, options={'maxiter': 300, 'disp': False})
                res_params = res.x
                
                # ALS refinement
                als_params = als_refine(res_params, R, max_iter=50)
                
                # Try snapping the ALS result
                result = snap_and_verify(als_params, R)
                if result:
                    best_result = result
                    found = True
                    break
                
                # Also try snapping the BFGS result
                result = snap_and_verify(res_params, R)
                if result:
                    best_result = result
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            return best_result

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
