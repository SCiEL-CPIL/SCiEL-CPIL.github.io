import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Precompute index arrays for vectorized objective
    # For each (i, j, k) triple, we need u_idx, v_idx, w_idx
    i_indices = []
    j_indices = []
    k_indices = []
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_indices.append(i)
                j_indices.append(j)
                k_indices.append(k)
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    
    i_indices = np.array(i_indices)
    j_indices = np.array(j_indices)
    k_indices = np.array(k_indices)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    num_triples = len(i_indices)
    
    # Helper to check validity
    def check_decomposition(U, V, W, rank):
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Check integrality (half-integer)
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-5): return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-5): return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-5): return False
        
        # Check correctness with multiple random matrices
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C = A @ B
            
            Cf = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for jj in range(m):
                        u_vec = U[ii*m+jj, :]
                        v_vec = V[jj*p+kk, :]
                        w_val = W[kk*n+ii, :]
                        temp = u_vec * v_vec
                        acc += np.dot(temp, w_val)
                    Cf[ii, kk] = acc
            
            if not np.allclose(C, Cf, atol=1e-5):
                return False
        return True
    
    def vectorized_objective(params, rank):
        """Vectorized objective function"""
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # For each triple, compute dot(U[u_idx] * V[v_idx], W[w_idx])
        U_sel = U_flat[u_indices]  # (num_triples, rank)
        V_sel = V_flat[v_indices]  # (num_triples, rank)
        W_sel = W_flat[w_indices]  # (num_triples, rank)
        
        # Element-wise multiply then dot with W
        prod = U_sel * V_sel  # (num_triples, rank)
        vals = np.sum(prod * W_sel, axis=1)  # (num_triples,)
        
        err = np.sum((1.0 - vals) ** 2)
        return err
    
    def vectorized_gradient(params, rank):
        """Gradient of the vectorized objective"""
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        U_sel = U_flat[u_indices]
        V_sel = V_flat[v_indices]
        W_sel = W_flat[w_indices]
        
        prod = U_sel * V_sel
        vals = np.sum(prod * W_sel, axis=1)
        residuals = 2.0 * (1.0 - vals)  # (num_triples,)
        
        # Gradient with respect to U
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        # d/dU: for each triple t, d/dU[u_idx, :] = -residuals[t] * V_sel[t] * W_sel[t]
        # Use np.add.at for accumulation
        for t in range(num_triples):
            grad_U[u_indices[t]] += -residuals[t] * V_sel[t] * W_sel[t]
            grad_V[v_indices[t]] += -residuals[t] * U_sel[t] * W_sel[t]
            grad_W[w_indices[t]] += -residuals[t] * U_sel[t] * V_sel[t]
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return grad_flat
    
    # Search for low rank
    best_result = None
    for rank in [16, 18, 20, 22, 24, 26, 27]:
        if time.time() > deadline: break
        found = False
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline: break
            
            # Varied initialization
            np.random.seed(rank * 1000 + restart * 7)
            scale = 0.3 + 0.2 * (restart % 3)
            params = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * scale
            
            try:
                res = minimize(
                    vectorized_objective,
                    params,
                    args=(rank,),
                    jac=lambda p, r: vectorized_gradient(p, r),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
                )
                
                if res.fun < 1e-6:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Try rounding to half-integers
                    U_int = np.round(2 * U_opt) / 2.0
                    V_int = np.round(2 * V_opt) / 2.0
                    W_int = np.round(2 * W_opt) / 2.0
                    
                    if check_decomposition(U_int, V_int, W_int, rank):
                        best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        found = True
                        break
                    
                    # If direct rounding fails, try a polish pass
                    if time.time() < deadline - 10:
                        # Local refinement around the rounded values
                        params_rounded = np.concatenate([
                            U_int.flatten(), V_int.flatten(), W_int.flatten()
                        ])
                        
                        res2 = minimize(
                            vectorized_objective,
                            params_rounded,
                            args=(rank,),
                            jac=lambda p, r: vectorized_gradient(p, r),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 3000}
                        )
                        
                        if res2.fun < 1e-6:
                            U_opt2 = res2.x[:n*m*rank].reshape(n*m, rank)
                            V_opt2 = res2.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                            W_opt2 = res2.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                            
                            U_int2 = np.round(2 * U_opt2) / 2.0
                            V_int2 = np.round(2 * V_opt2) / 2.0
                            W_int2 = np.round(2 * W_opt2) / 2.0
                            
                            if check_decomposition(U_int2, V_int2, W_int2, rank):
                                best_result = {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
                                found = True
                                break
            except Exception:
                pass
        
        if found:
            break

    if best_result:
        return best_result

    # Fallback to Strassen Rank 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
