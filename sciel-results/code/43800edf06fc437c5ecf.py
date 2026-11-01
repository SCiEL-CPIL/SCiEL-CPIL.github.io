import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback solution: Rank 21
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
    
    U_fb = [[0] * rank_fb for _ in range(4)]
    V_fb = [[0] * rank_fb for _ in range(12)]
    W_fb = [[0] * rank_fb for _ in range(12)]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U_fb[idx][col] = sc['U'][idx]
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V_fb[v_idx[idx]][col] = sc['V'][idx]
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W_fb[w_idx[idx]][col] = sc['W'][idx]
                    
    # Try to find rank 13 for M_{2,2,4}
    target_rank_4 = 13
    target_rank_6 = 20
    
    # Tensor for M_{2,2,4}
    T_4 = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                a = i*2 + j
                b = j*4 + k
                c = k*2 + i
                T_4[a, b, c] = 1.0

    def residuals_4(params):
        U_s = params[0:target_rank_4*4].reshape(target_rank_4, 4)
        V_s = params[target_rank_4*4:target_rank_4*12].reshape(target_rank_4, 8)
        W_s = params[target_rank_4*12:].reshape(target_rank_4, 8)
        
        S = np.einsum('sa,sb,sc->abc', U_s, V_s, W_s)
        
        return (S - T_4).flatten()

    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    def verify_solution_4(U, V, W):
        U_arr = np.array(U).T
        V_arr = np.array(V).T
        W_arr = np.array(W).T
        S = np.einsum('sa,sb,sc->abc', U_arr, V_arr, W_arr)
        return np.allclose(S, T_4, atol=1e-9)

    found_4 = False
    best_params_4 = None
    
    np.random.seed(42)
    
    # Search for rank 13
    for attempt in range(200):
        if time.time() > deadline - 40:
            break
            
        # Initialize
        # Try random
        init = np.random.randn(target_rank_4 * 20) * 0.5
        
        try:
            result = least_squares(residuals_4, init, method='lm', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if result.cost < 1e-10:
                U_4 = result.x[0:target_rank_4*4].reshape(target_rank_4, 4)
                V_4 = result.x[target_rank_4*4:target_rank_4*12].reshape(target_rank_4, 8)
                W_4 = result.x[target_rank_4*12:].reshape(target_rank_4, 8)
                
                # Snap to half-integers
                U_snap = snap_to_half_int(U_4)
                V_snap = snap_to_half_int(V_4)
                W_snap = snap_to_half_int(W_4)
                
                # Verify snapped solution
                if verify_solution_4(U_snap.T, V_snap.T, W_snap.T):
                    found_4 = True
                    best_params_4 = {
                        'U': U_snap.T.tolist(),
                        'V': V_snap.T.tolist(),
                        'W': W_snap.T.tolist()
                    }
                    break
        except Exception:
            pass

    if found_4 and best_params_4 is not None:
        U_4 = np.array(best_params_4['U'])
        V_4 = np.array(best_params_4['V'])
        W_4 = np.array(best_params_4['W'])
        
        # Construct Rank 20 for M_{2,2,6}
        U = [[0] * target_rank_6 for _ in range(4)]
        V = [[0] * target_rank_6 for _ in range(12)]
        W = [[0] * target_rank_6 for _ in range(12)]
        
        # Fill first 13 columns
        for s in range(target_rank_4):
            for idx in range(4):
                U[idx][s] = U_4[s, idx]
                
            for idx_4 in range(8):
                j = idx_4 // 4
                k = idx_4 % 4
                idx_6 = j*6 + k
                V[idx_6][s] = V_4[s, idx_4]
                
            for idx_4 in range(8):
                k = idx_4 // 2
                i = idx_4 % 2
                idx_6 = k*2 + i
                W[idx_6][s] = W_4[s, idx_4]
                
        # Fill last 7 columns with Strassen for block 3 (k=4,5)
        for s in range(7):
            col = target_rank_4 + s
            sc = strassen[s]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            v_idx = [4, 5, 10, 11]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
                    
            w_idx = [8, 9, 10, 11]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
                    
        # Verify final solution
        U_arr = np.array(U).T
        V_arr = np.array(V).T
        W_arr = np.array(W).T
        
        S = np.einsum('sa,sb,sc->abc', U_arr, V_arr, W_arr)
        
        T = np.zeros((4, 12, 12))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i*2 + j
                    b = j*6 + k
                    c = k*2 + i
                    T[a, b, c] = 1.0
                    
        if np.allclose(S, T, atol=1e-8):
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}
