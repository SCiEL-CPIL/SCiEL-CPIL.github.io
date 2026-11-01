import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # 1. Fallback: Rank 21 Solution
    # Strassen decomposition for 2x2 * 2x2, applied to 3 blocks of 2 columns.
    rank_fb = 21
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, 
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, 
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, 
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, 
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, 
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, 
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, 
    ]
    
    U_fb = np.zeros((4, rank_fb))
    V_fb = np.zeros((12, rank_fb))
    W_fb = np.zeros((12, rank_fb))
    
    for b in range(3): 
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            for idx in range(4):
                U_fb[idx, col] = sc['U'][idx]
            
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                V_fb[v_indices[idx], col] = sc['V'][idx]
            
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                W_fb[w_indices[idx], col] = sc['W'][idx]

    # 2. Attempt Rank 20
    target_rank = 20
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i*2 + j
                b = j*6 + k
                c = k*2 + i
                T[a, b, c] = 1.0

    def residuals(params):
        R = target_rank
        U_s = params[0:R*4].reshape(R, 4).T 
        V_s = params[R*4:R*16].reshape(R, 12).T 
        W_s = params[R*16:].reshape(R, 12).T
        
        S = np.einsum('as,bs,cs->abc', U_s, V_s, W_s)
        
        return (S - T).flatten()

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        S = np.einsum('as,bs,cs->abc', U_s, V_s, W_s)
        if np.allclose(S, T, atol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    found = False
    best_solution = None
    
    # Inits
    inits = []
    
    # Strategy 1: Truncated Rank-21
    for drop_col in range(21):
        mask = [i for i in range(21) if i != drop_col]
        U_init = U_fb[:, mask]
        V_init = V_fb[:, mask]
        W_init = W_fb[:, mask]
        
        init_params = np.concatenate([
            U_init.T.flatten(),
            V_init.T.flatten(),
            W_init.T.flatten()
        ])
        inits.append(init_params)
        
        if time.time() > deadline - 20:
            break

    # Strategy 2: Random Inits with different seeds
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        np.random.seed(seed)
        U_rand = np.random.randn(20, 4)
        V_rand = np.random.randn(20, 12)
        W_rand = np.random.randn(20, 12)
        
        # Scale to help convergence? Strassen entries are -1, 0, 1.
        # Let's try scaling by 1.
        init_params = np.concatenate([
            U_rand.T.flatten(),
            V_rand.T.flatten(),
            W_rand.T.flatten()
        ])
        inits.append(init_params)

    # Run optimization
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        try:
            result = least_squares(residuals, init, method='lm', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if result.cost < 1e-8: 
                R = target_rank
                U_sol = result.x[0:R*4].reshape(R, 4).T
                V_sol = result.x[R*4:R*16].reshape(R, 12).T
                W_sol = result.x[R*16:].reshape(R, 12).T
                
                valid, U_snap, V_snap, W_snap = snap_and_verify(U_sol, V_sol, W_sol)
                if valid:
                    found = True
                    best_solution = (U_snap, V_snap, W_snap)
                    break
        except Exception:
            continue

    if found:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
