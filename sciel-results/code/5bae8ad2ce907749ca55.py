import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # 1. Strassen for first 2 columns (k=0,1)
    # Rank 7
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((12, 7))
    W_strassen = np.zeros((12, 7))
    
    for s in range(7):
        step = strassen_steps[s]
        for l in range(4):
            U_strassen[l, s] = step['u'][l]
            b_global = [0, 1, 6, 7][l]
            V_strassen[b_global, s] = step['v'][l]
            c_global = [0, 2, 1, 3][l]
            W_strassen[c_global, s] = step['w'][l]

    # 2. Search for rank 13 decomposition of T_{2,2,4}
    T_224 = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k_prime in range(4):
                a = 2*i + j
                b = 2*j + k_prime
                c = 4*i + k_prime
                T_224[a, b, c] = 1.0
    
    def objective(params):
        U = params[:52].reshape(13, 4)
        V = params[52:156].reshape(13, 8)
        W = params[156:260].reshape(13, 8)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        res = T_calc - T_224
        return np.sum(res**2)

    def try_rank13(seed, max_iter=500):
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        
        # Define bounds for differential evolution
        # Entries are expected to be small integers or half-integers
        bounds = [(-3, 3)] * 260
        
        try:
            # Use differential evolution for global search
            sol = opt.differential_evolution(
                objective, 
                bounds, 
                seed=seed, 
                maxiter=max_iter, 
                tol=1e-10, 
                polish=True, 
                update_interval=10
            )
            
            if sol.fun < 1e-10:
                U_13 = sol.x[:52].reshape(13, 4)
                V_13 = sol.x[52:156].reshape(13, 8)
                W_13 = sol.x[156:260].reshape(13, 8)
                
                # Round to half-integers
                U_13_r = np.round(U_13 * 2) / 2
                V_13_r = np.round(V_13 * 2) / 2
                W_13_r = np.round(W_13 * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_13_r, V_13_r, W_13_r)
                if np.allclose(T_check, T_224, atol=1e-8):
                    return (U_13_r, V_13_r, W_13_r)
        except Exception:
            pass
        return None
    
    rank13_result = None
    # Try a few seeds with DE
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        rank13_result = try_rank13(seed)
        if rank13_result is not None:
            break
    
    if rank13_result is not None:
        U_13_r, V_13_r, W_13_r = rank13_result
        U = np.zeros((4, 20))
        V = np.zeros((12, 20))
        W = np.zeros((12, 20))
        
        U[:, :7] = U_strassen
        V[:, :7] = V_strassen
        W[:, :7] = W_strassen
        U[:, 7:20] = U_13_r
        
        for r in range(13):
            for lb in range(8):
                val = V_13_r[lb, r]
                if val != 0:
                    j = lb // 4
                    k_prime = lb % 4
                    k = 2 + k_prime
                    gb = 6*j + k
                    V[gb, 7+r] = val
                    
            for lc in range(8):
                val = W_13_r[lc, r]
                if val != 0:
                    i = lc // 4
                    k_prime = lc % 4
                    k = 2 + k_prime
                    gc = 2*k + i
                    W[gc, 7+r] = val
        
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        T_exp = np.zeros((4, 12, 12))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    T_exp[a, b, c] = 1.0
                    
        if np.allclose(T_full, T_exp, atol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Rank 21
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U[l, r] = step['u'][l]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V[b_map[l], r] = step['v'][l]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W[c_map[l], r] = step['w'][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
