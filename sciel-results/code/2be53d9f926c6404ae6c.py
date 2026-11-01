import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # 130s deadline, 20s safety margin
    
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

    # 2. Rank 13 for next 4 columns (k=2,3,4,5)
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
        return res.flatten()

    def try_rank13(x0, max_nfev=3000):
        try:
            sol = opt.least_squares(objective, x0, method='lm', max_nfev=max_nfev)
            if sol.cost < 1e-10:
                U_13 = sol.x[:52].reshape(13, 4)
                V_13 = sol.x[52:156].reshape(13, 8)
                W_13 = sol.x[156:260].reshape(13, 8)
                
                # Try rounding to integers and half-integers
                for scale in [1, 2, 4]:
                    U_r = np.round(U_13 * scale) / scale
                    V_r = np.round(V_13 * scale) / scale
                    W_r = np.round(W_13 * scale) / scale
                    T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_check, T_224, atol=1e-8):
                        return U_r, V_r, W_r
                # Also try without rounding if already very close
                T_check = np.einsum('ar,br,cr->abc', U_13, V_13, W_13)
                if np.allclose(T_check, T_224, atol=1e-8):
                    return U_13, V_13, W_13
        except Exception:
            pass
        return None

    # Build structured initial guesses
    # Strategy: T_{2,2,4} = sum over k' of (2x2 mult for column k')
    # Use 7 Strassen-like terms extended to all 4 k' values + 6 extra terms
    
    # Structured guess 1: Extend Strassen to cover all 4 k' values
    # For each Strassen step, V has entries at all 4 k' positions with same value
    # W has entries at all 4 k' positions with same value
    def make_structured_guess1():
        U = np.zeros((13, 4))
        V = np.zeros((13, 8))
        W = np.zeros((13, 8))
        for s in range(7):
            step = strassen_steps[s]
            for l in range(4):
                U[l, s] = step['u'][l]
            # V: local b = 2j+k', for each j in {0,1} and k' in {0,1,2,3}
            # v[0]=step['v'][0] (j=0), v[1]=step['v'][1] (j=0,k'=1)
            # Actually local b index: 2j+k'
            # j=0: b=0,1,2,3 for k'=0,1,2,3
            # j=1: b=4,5,6,7 for k'=0,1,2,3
            # step['v'][0] corresponds to j=0,k'=0 -> b=0
            # step['v'][1] corresponds to j=0,k'=1 -> b=1  
            # step['v'][2] corresponds to j=1,k'=0 -> b=4
            # step['v'][3] corresponds to j=1,k'=1 -> b=5
            # But we need all 4 k' values. The Strassen v only has 4 entries for 2 k' values.
            # For the extended version, each v entry should replicate across all 4 k'.
            # v_local[2j+k'] = step['v'][2j + (k' % 2)] ... this doesn't quite work
            # Let me just use the pattern: for j=0, all 4 k' get step['v'][0] or step['v'][1]
            # Actually, the original Strassen for 2x2 has B as 2x2 (j,k).
            # For 2x2x4, B is 2x4 (j,k'). The Strassen formula uses specific combinations.
            # Let's just initialize the extra 6 terms randomly and see.
            
            # For now, set V and W based on the 2x2 pattern replicated
            for k_prime in range(4):
                # b index for (j=0, k_prime) = k_prime
                # b index for (j=1, k_prime) = 4 + k_prime
                # c index for (i=0, k_prime) = k_prime
                # c index for (i=1, k_prime) = 4 + k_prime
                # Use the same v/w values as the 2x2 Strassen
                V[k_prime, s] = step['v'][0]   # j=0, use v[0]
                V[4 + k_prime, s] = step['v'][2]  # j=1, use v[2]
                W[k_prime, s] = step['w'][0]   # i=0, use w[0]
                W[4 + k_prime, s] = step['w'][2]  # i=1, use w[2]
        
        # Initialize remaining 6 terms randomly
        np.random.seed(123)
        U[7:, :] = np.random.randn(6, 4) * 0.5
        V[7:, :] = np.random.randn(6, 8) * 0.5
        W[7:, :] = np.random.randn(6, 8) * 0.5
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    # Try structured guess first
    if time.time() < DEADLINE:
        x0_struct = make_structured_guess1()
        result = try_rank13(x0_struct, max_nfev=5000)
        if result is not None:
            U_13_r, V_13_r, W_13_r = result
            # Construct full matrices
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
            # Verify
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            T_exp = np.zeros((4, 12, 12))
            for i in range(2):
                for j in range(2):
                    for k in range(6):
                        T_exp[i*2+j, j*6+k, k*2+i] = 1.0
            if np.allclose(T_full, T_exp, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try multiple random restarts
    np.random.seed(42)
    for attempt in range(20):
        if time.time() >= DEADLINE:
            break
        x0 = np.random.randn(260) * 0.5
        result = try_rank13(x0, max_nfev=2000)
        if result is not None:
            U_13_r, V_13_r, W_13_r = result
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
                        T_exp[i*2+j, j*6+k, k*2+i] = 1.0
            if np.allclose(T_full, T_exp, atol=1e-6):
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
