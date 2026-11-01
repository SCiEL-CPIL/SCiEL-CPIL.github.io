import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 36
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def verify_2x3x3(U, V, W, rank):
        """Verify rank decomposition for 2x3x3."""
        n_s, m_s, p_s = 2, 3, 3
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T = np.ones((n_s, m_s, p_s))
        T_approx = np.zeros((n_s, m_s, p_s))
        for r in range(rank):
            T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_approx, T, atol=1e-8)
    
    def find_2x3x3_rank11(max_trials=100):
        """Find rank-11 decomposition for 2x3x3."""
        n_s, m_s, p_s = 2, 3, 3
        T = np.ones((n_s, m_s, p_s))
        rank = 11
        
        for trial in range(max_trials):
            if time.time() > deadline - 30:
                return None
            
            # Use structured initialization based on Strassen-like patterns
            # U: 2x11, V: 3x11, W: 3x11 (W indexed by k for 2x3x3)
            
            # Try different random seeds
            np.random.seed(trial * 7 + 42)
            
            # Initialize with a mix of structured and random
            U = np.zeros((n_s, rank))
            V = np.zeros((m_s, rank))
            W = np.zeros((p_s, rank))
            
            # First 7 terms: Strassen-like for 2x2x2 core
            # Strassen terms for A[0:2,0:2] * B[0:2,0:2]
            # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
            U[0, 0] += 1; U[1, 0] += 1
            V[0, 0] += 1; V[1, 0] += 1
            W[0, 0] += 1; W[1, 0] += 1  # W indexed by k, but for 2x2x2, W[k,i]
            
            # Actually for 2x3x3, W is indexed by k (0..2) and i (0..1)
            # W[k, i] where the flat index is k*2+i
            
            # Let me use a different parameterization
            # U[i, r], V[j, r], W[k, r] where the tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            
            U = np.random.randn(n_s, rank) * 0.5
            V = np.random.randn(m_s, rank) * 0.5
            W = np.random.randn(p_s, rank) * 0.5
            
            def params_to_tensors(params):
                U = params[:n_s*rank].reshape(n_s, rank)
                V = params[n_s*rank:n_s*rank + m_s*rank].reshape(m_s, rank)
                W = params[n_s*rank + m_s*rank:].reshape(p_s, rank)
                return U, V, W
                
            def objective(params):
                U, V, W = params_to_tensors(params)
                T_approx = np.zeros((n_s, m_s, p_s))
                for r in range(rank):
                    T_approx += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                diff = T - T_approx
                return np.sum(diff**2)
                
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, initial_params, method='L-BFGS-B', maxiter=500, 
                                  options={'maxls': 100, 'gtol': 1e-8, 'ftol': 1e-12})
                if result.fun < 1e-10:
                    U, V, W = params_to_tensors(result.x)
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify_2x3x3(U_r, V_r, W_r, rank):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            except Exception:
                pass
        return None
    
    # Try to find rank-11 decomposition for 2x3x3
    decomp = find_2x3x3_rank11(max_trials=50)
    
    if decomp:
        U_sub, V_sub, W_sub = decomp
        R = 11
        total_rank = 2 * R  # 22
        
        U_full = [[0] * total_rank for _ in range(n * m)]  # 6 x 22
        V_full = [[0] * total_rank for _ in range(m * p)]  # 18 x 22
        W_full = [[0] * total_rank for _ in range(p * n)]  # 12 x 22
        
        # First block: k in 0..2
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][r] = U_sub[i][j]
                    
            for j in range(m):
                for k in range(3):
                    V_full[j * p + k][r] = V_sub[j][k]
                    
            for k in range(3):
                for i in range(n):
                    W_full[k * n + i][r] = W_sub[k][i]
                    
        # Second block: k in 3..5
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][R + r] = U_sub[i][j]
                    
            for j in range(m):
                for k in range(3):
                    V_full[j * p + (k + 3)][R + r] = V_sub[j][k]
                    
            for k in range(3):
                for i in range(n):
                    W_full[(k + 3) * n + i][R + r] = W_sub[k][i]
        
        # Verify full decomposition
        U_arr = np.array(U_full, dtype=float)
        V_arr = np.array(V_full, dtype=float)
        W_arr = np.array(W_full, dtype=float)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dec_val = 0.0
                    for r in range(total_rank):
                        dec_val += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                    if abs(dec_val - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = total_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full
        else:
            # Try without rounding
            U_full2 = [[0] * total_rank for _ in range(n * m)]
            V_full2 = [[0] * total_rank for _ in range(m * p)]
            W_full2 = [[0] * total_rank for _ in range(p * n)]
            
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        U_full2[i * m + j][r] = U_sub[i][j]
                    U_full2[i * m + j][R + r] = U_sub[i][j]
                for j in range(m):
                    for k in range(3):
                        V_full2[j * p + k][r] = V_sub[j][k]
                        V_full2[j * p + (k + 3)][R + r] = V_sub[j][k]
                for k in range(3):
                    for i in range(n):
                        W_full2[k * n + i][r] = W_sub[k][i]
                        W_full2[(k + 3) * n + i][R + r] = W_sub[k][i]
            
            U_arr = np.array(U_full2, dtype=float)
            V_arr = np.array(V_full2, dtype=float)
            W_arr = np.array(W_full2, dtype=float)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dec_val = 0.0
                        for r in range(total_rank):
                            dec_val += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                        if abs(dec_val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = total_rank
                best_U = U_full2
                best_V = V_full2
                best_W = W_full2
    
    return {"U": best_U, "V": best_V, "W": best_W}
