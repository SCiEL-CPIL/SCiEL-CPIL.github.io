import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
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
            # Map local B indices (0,1,2,3) to global B indices for k=0,1
            # local 0 -> j=0,k=0 -> b=0
            # local 1 -> j=0,k=1 -> b=1
            # local 2 -> j=1,k=0 -> b=6
            # local 3 -> j=1,k=1 -> b=7
            b_global = [0, 1, 6, 7][l]
            V_strassen[b_global, s] = step['v'][l]
            
            # Map local C indices (0,1,2,3) to global C indices for k=0,1
            # local 0 -> k=0,i=0 -> c=0
            # local 1 -> k=1,i=0 -> c=2
            # local 2 -> k=0,i=1 -> c=1
            # local 3 -> k=1,i=1 -> c=3
            c_global = [0, 2, 1, 3][l]
            W_strassen[c_global, s] = step['w'][l]

    # 2. Rank 13 for next 4 columns (k=2,3,4,5)
    # We need to find U, V, W of rank 13 for T_{2,2,4}
    # T_{2,2,4} tensor:
    # a = 2i+j
    # b = 2j+k' where k' in 0..3 corresponds to global k=2..5
    # c = 4i+k' where k' in 0..3 corresponds to global k=2..5
    
    # Global B indices for k=2..5:
    # k=2: b=2 (j=0), b=8 (j=1)
    # k=3: b=3 (j=0), b=9 (j=1)
    # k=4: b=4 (j=0), b=10 (j=1)
    # k=5: b=5 (j=0), b=11 (j=1)
    
    # Global C indices for k=2..5:
    # k=2: c=4 (i=0), c=5 (i=1)
    # k=3: c=6 (i=0), c=7 (i=1)
    # k=4: c=8 (i=0), c=9 (i=1)
    # k=5: c=10 (i=0), c=11 (i=1)
    
    T_224 = np.zeros((4, 8, 8)) # Local indices for 2x2x4
    # Local a: 0..3
    # Local b: 0..7 (j*4 + k')
    # Local c: 0..7 (i*4 + k')
    for i in range(2):
        for j in range(2):
            for k_prime in range(4):
                a = 2*i + j
                b = 2*j + k_prime
                c = 4*i + k_prime
                T_224[a, b, c] = 1.0

    # Search for rank 13 decomposition
    # Variables: 13 * (4 + 8 + 8) = 260
    # Residual: 320
    
    def objective(params):
        U = params[:52].reshape(13, 4)
        V = params[52:156].reshape(13, 8)
        W = params[156:260].reshape(13, 8)
        
        # Compute tensor
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residual
        res = T_calc - T_224
        return res.flatten()

    # Initial guess: Random
    np.random.seed(42)
    x0 = np.random.randn(260)
    
    # Optimize
    try:
        sol = opt.least_squares(objective, x0, method='lm', max_nfev=5000)
        
        if sol.cost < 1e-8:
            U_13 = sol.x[:52].reshape(13, 4)
            V_13 = sol.x[52:156].reshape(13, 8)
            W_13 = sol.x[156:260].reshape(13, 8)
            
            # Round to nearest half-integer or integer if close
            # The coefficients are likely integers or half-integers
            # We check if rounding works
            U_13_r = np.round(U_13 * 2) / 2
            V_13_r = np.round(V_13 * 2) / 2
            W_13_r = np.round(W_13 * 2) / 2
            
            T_check = np.einsum('ar,br,cr->abc', U_13_r, V_13_r, W_13_r)
            if np.allclose(T_check, T_224, atol=1e-6):
                # Construct full U, V, W
                U = np.zeros((4, 20))
                V = np.zeros((12, 20))
                W = np.zeros((12, 20))
                
                # Copy Strassen
                U[:, :7] = U_strassen
                V[:, :7] = V_strassen
                W[:, :7] = W_strassen
                
                # Copy Rank 13
                U[:, 7:20] = U_13_r
                
                # Map V_13 (local b) to global V
                # local b = 2j + k'
                # global b = 6j + k where k = 2 + k'
                for r in range(13):
                    for lb in range(8):
                        val = V_13_r[lb, r]
                        if val != 0:
                            j = lb // 4
                            k_prime = lb % 4
                            k = 2 + k_prime
                            gb = 6*j + k
                            V[gb, 7+r] = val
                            
                # Map W_13 (local c) to global W
                # local c = 4i + k'
                # global c = 2k + i where k = 2 + k'
                for r in range(13):
                    for lc in range(8):
                        val = W_13_r[lc, r]
                        if val != 0:
                            i = lc // 4
                            k_prime = lc % 4
                            k = 2 + k_prime
                            gc = 2*k + i
                            W[gc, 7+r] = val
                
                # Final Verification
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                T_exp = np.zeros((4, 12, 12))
                for i in range(2):
                    for j in range(2):
                        for k in range(6):
                            a = i * 2 + j
                            b = j * 6 + k
                            c = k * 2 + i
                            T_exp[a, b, c] = 1.0
                            
                if np.allclose(T_full, T_exp, atol=1e-6):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    except Exception as e:
        pass
        
    # Fallback to Rank 21
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    # Fill Strassen for 3 blocks
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U[l, r] = step['u'][l]
                
                # Map B
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V[b_map[l], r] = step['v'][l]
                
                # Map C
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W[c_map[l], r] = step['w'][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
