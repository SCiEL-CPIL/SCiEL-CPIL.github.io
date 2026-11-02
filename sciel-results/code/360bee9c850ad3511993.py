import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    n, m, p = 2, 2, 6
    nm, mp, pn = 4, 12, 12
    
    # Target tensor T_exp for M_2,2,6
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = 2*i + j
                b = 6*j + k
                c = 2*k + i
                T_exp[a, b, c] = 1.0

    # Fallback Rank 21
    def get_rank21():
        U = np.zeros((4, 21))
        V = np.zeros((12, 21))
        W = np.zeros((12, 21))
        
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        
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
        return U, V, W

    # Search for Rank 10 decomposition of M_2,2,3
    # T_223[a, b, c] = 1 if a=2i+j, b=3j+k, c=2k+i
    T_223 = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = 2*i + j
                b = 3*j + k
                c = 2*k + i
                T_223[a, b, c] = 1.0

    def objective_223(params):
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        res = T_calc - T_223
        return res.flatten()

    def try_rank10_223(x0, max_nfev=2000):
        try:
            sol = opt.least_squares(objective_223, x0, method='lm', max_nfev=max_nfev)
            if sol.cost < 1e-10:
                U_10 = sol.x[:40].reshape(4, 10)
                V_10 = sol.x[40:100].reshape(6, 10)
                W_10 = sol.x[100:160].reshape(6, 10)
                
                # Check rounding
                for scale in [1, 2]:
                    U_r = np.round(U_10 * scale) / scale
                    V_r = np.round(V_10 * scale) / scale
                    W_r = np.round(W_10 * scale) / scale
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T_223, atol=1e-8):
                        return U_r, V_r, W_r
                if np.allclose(np.einsum('ar,br,cr->abc', U_10, V_10, W_10), T_223, atol=1e-8):
                    return U_10, V_10, W_10
        except Exception:
            pass
        return None

    # Try to find rank 10 for M_2,2,3
    found_223 = None
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 12345, 54321, 13579, 24680]:
        if time.time() > DEADLINE - 10: break
        np.random.seed(seed)
        x0 = np.random.randn(160) * 0.5
        res = try_rank10_223(x0)
        if res is not None:
            found_223 = res
            break

    if found_223 is not None:
        U_10, V_10, W_10 = found_223
        # Construct Rank 20 for M_2,2,6
        U = np.zeros((4, 20))
        V = np.zeros((12, 20))
        W = np.zeros((12, 20))
        
        # Block 1: k=0,1,2
        U[:, :10] = U_10
        for r in range(10):
            for lb in range(6):
                j = lb // 3
                k = lb % 3
                gb = 6*j + k
                V[gb, r] = V_10[lb, r]
            for lc in range(6):
                i = lc % 2
                k = lc // 2
                gc = 2*k + i
                W[gc, r] = W_10[lc, r]
                
        # Block 2: k=3,4,5
        U[:, 10:20] = U_10
        for r in range(10):
            for lb in range(6):
                j = lb // 3
                k_prime = lb % 3
                k = k_prime + 3
                gb = 6*j + k
                V[gb, 10+r] = V_10[lb, r]
            for lc in range(6):
                i = lc % 2
                k_prime = lc // 2
                k = k_prime + 3
                gc = 2*k + i
                W[gc, 10+r] = W_10[lc, r]
                
        # Verify
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_full, T_exp, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 21
    U, V, W = get_rank21()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
