"""Matrix multiplication 2x3x8 rank-minimal exact decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n_sub, m_sub, p_sub = n, m, p // 2  # 2, 3, 4
    
    # Helper to build the full 2x3x8 decomposition from a 2x3x4 sub-decomposition
    def build_from_sub(U_sub, V_sub, W_sub, n, m, p):
        r_sub = U_sub.shape[1]
        r_total = 2 * r_sub
        U = np.zeros((n*m, r_total))
        V = np.zeros((m*p, r_total))
        W = np.zeros((p*n, r_total))
        
        # U is the same for both blocks
        U[:, :r_sub] = U_sub
        U[:, r_sub:] = U_sub
        
        # V is stacked
        # V_sub has shape (m*p_sub, r_sub) = (3*4, r_sub) = (12, r_sub)
        # V has shape (m*p, r_total) = (3*8, 2*r_sub) = (24, 2*r_sub)
        # Block 0 (k=0..3) uses V_sub
        # Block 1 (k=4..7) uses V_sub
        V[:m*p_sub, :r_sub] = V_sub
        V[m*p_sub:, r_sub:] = V_sub
        
        # W is stacked/mapped
        # W_sub has shape (p_sub*n, r_sub) = (4*2, r_sub) = (8, r_sub)
        # W has shape (p*n, r_total) = (8*2, 2*r_sub) = (16, 2*r_sub)
        # W index: k*n + i
        # For k in 0..3 (block 0), row k*n+i in W corresponds to k_sub=k in W_sub
        # For k in 4..7 (block 1), row k*n+i in W corresponds to k_sub=k-4 in W_sub
        for k in range(p_sub):
            W[k*n:(k+1)*n, :r_sub] = W_sub[k*n:(k+1)*n, :]
        for k in range(p_sub):
            W[(k+p_sub)*n:(k+p_sub+1)*n, r_sub:] = W_sub[k*n:(k+1)*n, :]
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Helper to verify the full decomposition
    def verify_full(U, V, W, n, m, p, tol=1e-5):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Search for a low-rank decomposition of the 2x3x4 tensor
    def find_sub_decomposition(n, m, p, target_rank, timeout=10):
        # Objective function
        def objective(params):
            # Unpack U, V, W
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            size_W = p * n * target_rank
            
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (s - 1.0)**2
            return error

        # Initial guess
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(n*m*target_rank + m*p*target_rank + p*n*target_rank) * 0.1
        
        start_time = time.time()
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'disp': False})
            if time.time() - start_time > timeout:
                return None
            if res.fun < 1e-6:
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                U = res.x[:size_U].reshape((n*m, target_rank))
                V = res.x[size_U:size_U+size_V].reshape((m*p, target_rank))
                W = res.x[size_U+size_V:].reshape((p*n, target_rank))
                return U, V, W
        except Exception:
            pass
        return None

    # Try ranks from 13 up to 23 for the sub-problem
    best_sub = None
    best_rank = 24 # Schoolbook
    
    # Schoolbook for sub-problem
    U_sub_school = np.zeros((n_sub*m_sub, n_sub*m_sub*p_sub))
    V_sub_school = np.zeros((m_sub*p_sub, n_sub*m_sub*p_sub))
    W_sub_school = np.zeros((p_sub*n_sub, n_sub*m_sub*p_sub))
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U_sub_school[i*m_sub+j, col] = 1
                V_sub_school[j*p_sub+k, col] = 1
                W_sub_school[k*n_sub+i, col] = 1
                col += 1
    best_sub = (U_sub_school, V_sub_school, W_sub_school)
    
    for r in range(13, 24):
        if time.time() > 110: # Safety margin
            break
        sub = find_sub_decomposition(n_sub, m_sub, p_sub, r)
        if sub is not None:
            best_sub = sub
            best_rank = r
            break # Found the lowest rank, stop
    
    # Construct final result
    U_final, V_final, W_final = build_from_sub(best_sub[0], best_sub[1], best_sub[2], n, m, p)
    
    # Verify
    if not verify_full(U_final, V_final, W_final, n, m, p):
        # Fallback to full schoolbook
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
        
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
