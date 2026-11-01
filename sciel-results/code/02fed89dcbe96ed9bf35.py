import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 100.0  # Leave margin for 150s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index arrays for vectorized error computation
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    u_idx = (i_vals * m + j_vals).flatten()
    v_idx = (j_vals * p + k_vals).flatten()
    w_idx = (k_vals * n + i_vals).flatten()
    num_terms = n * m * p

    def error_and_grad(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp :]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        # Vectorized: for each term, compute sum over rank
        # U[u_idx, :], V[v_idx, :], W[w_idx, :] are (num_terms, r)
        U_sel = U[:, u_idx].T  # (num_terms, r)
        V_sel = V[:, v_idx].T  # (num_terms, r)
        W_sel = W[:, w_idx].T  # (num_terms, r)
        
        # Product along rank axis
        prod = U_sel * V_sel * W_sel  # (num_terms, r)
        sums = prod.sum(axis=1)  # (num_terms,)
        
        err = np.sum((sums - 1.0) ** 2)
        
        # Gradients
        diff = (sums - 1.0)[:, np.newaxis]  # (num_terms, 1)
        
        # dE/dU_sel = 2 * diff * V_sel * W_sel
        dU_sel = 2 * diff * V_sel * W_sel  # (num_terms, r)
        dV_sel = 2 * diff * U_sel * W_sel  # (num_terms, r)
        dW_sel = 2 * diff * U_sel * V_sel  # (num_terms, r)
        
        # Accumulate gradients back to full U, V, W
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # np.add.at is slow, use bincount or advanced indexing accumulation
        # Since indices are small, we can just loop over terms if num_terms is small, 
        # but vectorized scatter is better.
        # For simplicity and speed with small num_terms (24), we can use loops or 
        # a trick with np.add.at.
        
        # Actually, for num_terms=24 and r~23, the arrays are small.
        # Let's just do the accumulation with loops for clarity and avoid complex scatter bugs, 
        # or use np.add.at which is correct.
        
        np.add.at(dU, u_idx, dU_sel.T) # dU_sel.T is (r, num_terms), u_idx is (num_terms,)
        # Wait, dU is (r, nm). u_idx maps term -> index in nm.
        # We want dU[c, u_idx[t]] += dU_sel[t, c]
        # So dU_sel.T is (r, num_terms).
        # np.add.at(dU, (slice(None), u_idx), dU_sel.T)
        
        # Let's do it properly with explicit loops for accumulation to be safe and fast enough
        dU = np.zeros((r, nm))
        dV = np.zeros((r, mp))
        dW = np.zeros((r, pn))
        
        for t in range(num_terms):
            dU[:, u_idx[t]] += dU_sel[t]
            dV[:, v_idx[t]] += dV_sel[t]
            dW[:, w_idx[t]] += dW_sel[t]
            
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        return float(err), grad

    def check_solution(U, V, W, r):
        for c in range(r):
            for val in U[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in V[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
            for val in W[c]:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx_val = i * m + j
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx_val] * V[c, v_idx_val] * W[c, w_idx_val]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_and_return(U, V, W, r):
        U_list = [[round(val * 2) / 2.0 for val in row] for row in U.tolist()]
        V_list = [[round(val * 2) / 2.0 for val in row] for row in V.tolist()]
        W_list = [[round(val * 2) / 2.0 for val in row] for row in W.tolist()]
        if check_solution(np.array(U_list), np.array(V_list), np.array(W_list), r):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    def polish(x0, r, max_iter=1000):
        """Polish using L-BFGS-B with gradient."""
        from scipy.optimize import minimize
        
        try:
            res = minimize(error_and_grad, x0, args=(r,), method='L-BFGS-B', 
                           jac=True,
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            return None
        
        if res.fun > 1e-6:
            return None
        
        return res.x

    # Try ranks from 23 down to 18
    for r in range(rank_sb - 1, 17, -1):
        if time.time() - start_time > max_time:
            break
            
        total_vars = r * (nm + mp + pn)
        max_attempts = 10 if r == 23 else 20
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break
            
            # Random initialization
            x0 = np.random.randn(total_vars) * 0.5
            
            # Sometimes try schoolbook-like initialization with noise
            if attempt % 3 == 0:
                x0 = np.zeros(total_vars)
                for c in range(min(r, rank_sb)):
                    i, jk = divmod(c, m * p)
                    j, k = divmod(jk, p)
                    x0[c * nm + i * m + j] = 1.0
                    x0[r * nm + c * mp + j * p + k] = 1.0
                    x0[r * nm + r * mp + c * pn + k * n + i] = 1.0
                x0 += 0.1 * np.random.randn(total_vars)
            
            x_result = polish(x0, r)
            if x_result is not None:
                U_flat = x_result[:r * nm]
                V_flat = x_result[r * nm : r * nm + r * mp]
                W_flat = x_result[r * nm + r * mp :]
                
                U = U_flat.reshape(r, nm)
                V = V_flat.reshape(r, mp)
                W = W_flat.reshape(r, pn)
                
                result = verify_and_return(U, V, W, r)
                if result is not None:
                    return result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
