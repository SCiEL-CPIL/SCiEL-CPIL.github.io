import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1] # Wait, B index is j*6+k.
            # For block b (k=2b, 2b+1):
            # j=0: k=2b, 2b+1 -> b_idx = 0*6 + 2b, 0*6 + 2b+1 = 2b, 2b+1
            # j=1: k=2b, 2b+1 -> b_idx = 1*6 + 2b, 1*6 + 2b+1 = 6+2b, 6+2b+1
            # So v_idx is correct for the 4 entries of the 2x2 block.
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V_fb[v_idx[idx]][col] = sc['V'][idx]
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3] # W index is k*2+i.
            # For block b (k=2b, 2b+1):
            # k=2b: c = 2(2b)+i = 4b, 4b+1
            # k=2b+1: c = 2(2b+1)+i = 4b+2, 4b+3
            # So w_idx is correct.
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W_fb[w_idx[idx]][col] = sc['W'][idx]
                    
    # Try to find rank 13 for M_{2,2,4}
    target_rank_4 = 13
    target_rank_6 = 20
    
    # Tensor for M_{2,2,4}
    # A: 2x2 (4)
    # B: 2x4 (8)
    # C: 2x4 (8)
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
        
    found_4 = False
    best_params_4 = None
    
    np.random.seed(42)
    
    # Try to find rank 13
    for _ in range(100):
        if time.time() > deadline - 30:
            break
            
        init = np.random.randn(target_rank_4 * 20) * 0.1
        # Add some structure? Maybe random integer-like?
        
        try:
            result = least_squares(residuals_4, init, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.cost < 1e-10:
                found_4 = True
                best_params_4 = result.x
                break
        except Exception:
            pass
            
    if found_4 and best_params_4 is not None:
        U_4 = best_params_4[0:target_rank_4*4].reshape(target_rank_4, 4)
        V_4 = best_params_4[target_rank_4*4:target_rank_4*12].reshape(target_rank_4, 8)
        W_4 = best_params_4[target_rank_4*12:].reshape(target_rank_4, 8)
        
        # Construct Rank 20 for M_{2,2,6}
        U = [[0] * target_rank_6 for _ in range(4)]
        V = [[0] * target_rank_6 for _ in range(12)]
        W = [[0] * target_rank_6 for _ in range(12)]
        
        # Fill first 13 columns
        for s in range(target_rank_4):
            for idx in range(4):
                U[idx][s] = U_4[s, idx]
                
            # Map V_4 indices (0..7) to V_6 indices (0..11)
            # V_4 b = j*4 + k, j in {0,1}, k in {0..3}
            # V_6 b = j*6 + k, j in {0,1}, k in {0..5}
            # So b_4 = j*4 + k maps to b_6 = j*6 + k
            for idx_4 in range(8):
                j = idx_4 // 4
                k = idx_4 % 4
                idx_6 = j*6 + k
                V[idx_6][s] = V_4[s, idx_4]
                
            # Map W_4 indices (0..7) to W_6 indices (0..11)
            # W_4 c = k*2 + i, k in {0..3}, i in {0,1}
            # W_6 c = k*2 + i, k in {0..5}, i in {0,1}
            # So c_4 = k*2 + i maps to c_6 = k*2 + i (same formula, just k range differs)
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
            
            # Block 3: k=4,5
            # j=0: k=4,5 -> b = 0*6+4=4, 0*6+5=5
            # j=1: k=4,5 -> b = 1*6+4=10, 1*6+5=11
            v_idx = [4, 5, 10, 11]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
                    
            # k=4,5
            # k=4: c = 4*2+i = 8, 9
            # k=5: c = 5*2+i = 10, 11
            w_idx = [8, 9, 10, 11]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
                    
        # Verify solution
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
