import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 5
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute indices for the 30 non-zero entries of the tensor
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def objective(params):
        R = len(params) // (dim_u + dim_v + dim_w)
        U = params[0:R*dim_u].reshape(R, dim_u)
        V = params[R*dim_u:R*dim_u+R*dim_v].reshape(R, dim_v)
        W = params[R*dim_u+R*dim_v:R*dim_u+R*dim_v+R*dim_w].reshape(R, dim_w)
        
        # Vectorized computation of the 30 values
        U_sel = U[:, u_indices] # (R, 30)
        V_sel = V[:, v_indices] # (R, 30)
        W_sel = W[:, w_indices] # (R, 30)
        
        # Sum over R
        vals = np.sum(U_sel * V_sel * W_sel, axis=0) # (30,)
        
        error = np.sum((vals - 1.0) ** 2)
        return error

    # Try rank 27
    for rank in [27, 28]:
        np.random.seed(42)
        best_params = None
        best_error = float('inf')
        
        for restart in range(10):
            x0 = np.random.randn(rank * (dim_u + dim_v + dim_w))
            # Scale init?
            # Trivial terms are 1.
            # Random normal is okay.
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
        
        if best_error < 1e-6:
            R = rank
            U = best_params[0:R*dim_u].reshape(R, dim_u)
            V = best_params[R*dim_u:R*dim_u+R*dim_v].reshape(R, dim_v)
            W = best_params[R*dim_u+R*dim_v:R*dim_u+R*dim_v+R*dim_w].reshape(R, dim_w)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            U_sel = U_r[:, u_indices]
            V_sel = V_r[:, v_indices]
            W_sel = W_r[:, w_indices]
            vals = np.sum(U_sel * V_sel * W_sel, axis=0)
            
            if np.allclose(vals, 1.0, atol=1e-9):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    # Fallback: Return the analytic 28 rank decomposition
    # (Code from current program)
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    strassen_steps = [
        ({1: 1, 5: 1}, {5: 1, 11: 1}, {0: 1, 3: 1}),
        ({4: 1, 5: 1}, {5: 1}, {1: 1, 3: -1}),
        ({1: 1}, {6: 1, 11: -1}, {2: 1, 3: 1}),
        ({5: 1}, {10: 1, 5: -1}, {0: 1, 1: 1}),
        ({1: 1, 2: 1}, {11: 1}, {0: -1, 2: 1}),
        ({4: 1, 1: -1}, {5: 1, 6: 1}, {3: 1}),
        ({2: 1, 5: -1}, {10: 1, 11: 1}, {0: 1})
    ]
    for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
        for idx, val in u_coeffs.items(): U[idx, col] = val
        for idx, val in v_coeffs.items(): V[idx, col] = val
        for idx, val in w_coeffs.items(): W[idx, col] = val
        col += 1
        
    strassen_steps_3a = [
        ({1: 1, 5: 1}, {7: 1, 13: 1}, {4: 1, 7: 1}),
        ({4: 1, 5: 1}, {7: 1}, {5: 1, 7: -1}),
        ({1: 1}, {8: 1, 13: -1}, {6: 1, 7: 1}),
        ({5: 1}, {12: 1, 7: -1}, {4: 1, 5: 1}),
        ({1: 1, 2: 1}, {13: 1}, {4: -1, 6: 1}),
        ({4: 1, 1: -1}, {7: 1, 8: 1}, {7: 1}),
        ({2: 1, 5: -1}, {12: 1, 13: 1}, {4: 1})
    ]
    for u_coeffs, v_coeffs, w_coeffs in strassen_steps_3a:
        for idx, val in u_coeffs.items(): U[idx, col] = val
        for idx, val in v_coeffs.items(): V[idx, col] = val
        for idx, val in w_coeffs.items(): W[idx, col] = val
        col += 1
        
    U[1, col] = 1; V[9, col] = 1; W[8, col] = 1; col += 1
    U[2, col] = 1; V[14, col] = 1; W[8, col] = 1; col += 1
    U[4, col] = 1; V[9, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[14, col] = 1; W[9, col] = 1; col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
