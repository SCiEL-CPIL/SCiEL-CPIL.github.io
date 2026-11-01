import numpy as np
import time
import scipy.optimize as opt

def construct():
    t_start = time.time()
    deadline = t_start + 120.0  # 120s limit, use 110s budget for safety
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Strassen steps for 2x2 matrix multiplication
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Helper to build Strassen block for columns k0..k0+1
    def build_strassen_block(k0):
        U_s = np.zeros((4, 7))
        V_s = np.zeros((12, 7))
        W_s = np.zeros((12, 7))
        for s in range(7):
            step = strassen_steps[s]
            for l in range(4):
                U_s[l, s] = step['u'][l]
                k1 = k0 + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V_s[b_map[l], s] = step['v'][l]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W_s[c_map[l], s] = step['w'][l]
        return U_s, V_s, W_s
    
    # Build target tensor
    T_full_target = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_full_target[a, b, c] = 1.0
    
    # Try rank 20: Strassen for k=0,1 (rank 7) + rank 13 for k=2,3,4,5
    # Local tensor for 2x2x4
    T_224 = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k_prime in range(4):
                a = 2*i + j
                b = 2*j + k_prime
                c = 4*i + k_prime
                T_224[a, b, c] = 1.0
    
    def objective_r13(params):
        U = params[:52].reshape(13, 4)
        V = params[52:156].reshape(13, 8)
        W = params[156:260].reshape(13, 8)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        res = T_calc - T_224
        return res.flatten()
    
    # Try multiple random restarts for rank 13
    best_sol_r13 = None
    best_cost_r13 = np.inf
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        x0 = np.random.randn(260) * 0.5
        try:
            sol = opt.least_squares(objective_r13, x0, method='lm', max_nfev=3000)
            if sol.cost < best_cost_r13:
                best_cost_r13 = sol.cost
                best_sol_r13 = sol.x.copy()
                if sol.cost < 1e-8:
                    break
        except Exception:
            continue
    
    # Check if rank 20 works
    if best_sol_r13 is not None and best_cost_r13 < 1e-6:
        U_13 = best_sol_r13[:52].reshape(13, 4)
        V_13 = best_sol_r13[52:156].reshape(13, 8)
        W_13 = best_sol_r13[156:260].reshape(13, 8)
        
        # Try rounding to half-integers
        for mult in [1, 2, 4]:
            U_r = np.round(U_13 * mult) / mult
            V_r = np.round(V_13 * mult) / mult
            W_r = np.round(W_13 * mult) / mult
            
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T_224, atol=1e-6):
                # Build full rank 20 decomposition
                U_s, V_s, W_s = build_strassen_block(0)
                
                U = np.zeros((4, 20))
                V = np.zeros((12, 20))
                W = np.zeros((12, 20))
                
                U[:, :7] = U_s
                V[:, :7] = V_s
                W[:, :7] = W_s
                
                U[:, 7:20] = U_r
                
                # Map V_13 local b to global
                for r in range(13):
                    for lb in range(8):
                        val = V_r[lb, r]
                        if abs(val) > 1e-10:
                            j = lb // 4
                            k_prime = lb % 4
                            k = 2 + k_prime
                            gb = 6*j + k
                            V[gb, 7+r] = val
                            
                # Map W_13 local c to global
                for r in range(13):
                    for lc in range(8):
                        val = W_r[lc, r]
                        if abs(val) > 1e-10:
                            i = lc // 4
                            k_prime = lc % 4
                            k = 2 + k_prime
                            gc = 2*k + i
                            W[gc, 7+r] = val
                
                # Verify
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_calc, T_full_target, atol=1e-6):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Rank 21 using 3 Strassen blocks
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for g in range(3):
        U_s, V_s, W_s = build_strassen_block(2*g)
        U[:, 7*g:(g+1)*7] = U_s
        V[:, 7*g:(g+1)*7] = V_s
        W[:, 7*g:(g+1)*7] = W_s
    
    # Verify rank 21
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(T_calc, T_full_target, atol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Should not reach here, but return anyway
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
