import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts to find a rank 20 decomposition.
    Falls back to rank 21 if rank 20 is not found within the time limit.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the rank 21 Strassen solution
    rank_21 = 21
    U_21 = np.zeros((nm, rank_21))
    V_21 = np.zeros((mp, rank_21))
    W_21 = np.zeros((pn, rank_21))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0: return 0 * 6 + k0
        elif local_idx == 1: return 0 * 6 + k1
        elif local_idx == 2: return 1 * 6 + k0
        else: return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0: return k0 * 2 + 0
        elif local_idx == 1: return k1 * 2 + 0
        elif local_idx == 2: return k0 * 2 + 1
        else: return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U_21[map_a(local_idx), r] = u_val
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V_21[map_b(local_idx, g), r] = v_val
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W_21[map_c(local_idx, g), r] = w_val

    # Expected tensor
    T_exp = np.zeros((nm, mp, pn))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0

    def residual(params):
        rank = len(params) // 28
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+12*rank].reshape(12, rank)
        W = params[4*rank+12*rank:].reshape(12, rank)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_calc - T_exp).flatten()

    # Try to find rank 20
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_result = None
    best_res_norm = np.inf
    
    np.random.seed(42)
    
    rank = 20
    num_params = 28 * rank
    
    # Strategy: multiple random restarts with varying initialization
    # 1) From rank-21 truncated + noise
    # 2) Pure random
    # 3) Structured random (sparse)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        if attempt < 10:
            # Initialize from truncated rank-21 with noise
            U_init = U_21[:, :20].copy()
            V_init = V_21[:, :20].copy()
            W_init = W_21[:, :20].copy()
            noise_scale = 0.1 * (1 + attempt)
            U_init += np.random.normal(0, noise_scale, U_init.shape)
            V_init += np.random.normal(0, noise_scale, V_init.shape)
            W_init += np.random.normal(0, noise_scale, W_init.shape)
        elif attempt < 30:
            # Pure random initialization
            U_init = np.random.normal(0, 1, (4, rank))
            V_init = np.random.normal(0, 1, (12, rank))
            W_init = np.random.normal(0, 1, (12, rank))
        else:
            # Sparse random initialization
            U_init = np.zeros((4, rank))
            V_init = np.zeros((12, rank))
            W_init = np.zeros((12, rank))
            for r_idx in range(rank):
                # Each rank-1 component touches a few entries
                u_nonzero = np.random.choice(4, size=np.random.randint(1, 4), replace=False)
                v_nonzero = np.random.choice(12, size=np.random.randint(1, 4), replace=False)
                w_nonzero = np.random.choice(12, size=np.random.randint(1, 4), replace=False)
                for idx in u_nonzero:
                    U_init[idx, r_idx] = np.random.choice([-1, 1])
                for idx in v_nonzero:
                    V_init[idx, r_idx] = np.random.choice([-1, 1])
                for idx in w_nonzero:
                    W_init[idx, r_idx] = np.random.choice([-1, 1])
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = least_squares(residual, x0, method='trf', max_nfev=2000, xtol=1e-8, ftol=1e-8)
            res_norm = np.linalg.norm(res.fun)
            
            if res_norm < best_res_norm:
                best_res_norm = res_norm
                best_result = res.x.copy()
            
            if res_norm < 1e-7:
                break
                
        except Exception:
            continue

    if best_result is not None and best_res_norm < 1e-7:
        rank = 20
        U = best_result[:4*rank].reshape(4, rank)
        V = best_result[4*rank:4*rank+12*rank].reshape(12, rank)
        W = best_result[4*rank+12*rank:].reshape(12, rank)
        
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_check, T_exp, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 21
    return {"U": U_21.tolist(), "V": V_21.tolist(), "W": W_21.tolist()}
