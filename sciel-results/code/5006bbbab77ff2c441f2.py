import numpy as np
import time
import itertools
import scipy.optimize

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    T_target = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, j, k, i, k] = 1.0
    T_target_flat = T_target.flatten()
    
    def residual(params):
        R = len(params) // (nm + mp + pn)
        if R == 0:
            return np.zeros(len(T_target_flat))
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        A = np.einsum('ijr,jkr->ijjkr', U_3d, V_3d)
        T_calc = np.einsum('ijjkr,kir->ijjkii', A, W_3d)
        return T_calc.flatten() - T_target_flat

    def try_rank(R, max_restarts=10, timeout=10):
        start_time = time.time()
        for _ in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            params = np.random.randn(R * (nm + mp + pn)) * 0.1
            if R > 0:
                params[0] = 1.0
            try:
                result = scipy.optimize.least_squares(residual, params, method='lm', max_nfev=500)
                if np.linalg.norm(result.fun) < 1e-5:
                    U = result.x[:nm*R].reshape(nm, R)
                    V = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W = result.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                    res_r = residual(params_r)
                    if np.linalg.norm(res_r) < 1e-5:
                        return U_r, V_r, W_r
            except Exception:
                pass
        return None

    # Fallback: Rank 39 decomposition
    def get_rank_39():
        terms = []
        def add_term(u, v, w):
            u_list = [0.0] * (n * m)
            v_list = [0.0] * (m * p)
            w_list = [0.0] * (p * n)
            for idx, val in u.items(): u_list[idx] = val
            for idx, val in v.items(): v_list[idx] = val
            for idx, val in w.items(): w_list[idx] = val
            terms.append((u_list, v_list, w_list))
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
        add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
        
        # Part 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2]
        add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
        add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
        add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,2): 1}, {w_idx(2,1): 1})
        
        # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 3:5]
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1, w_idx(4,1): 1})
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1, w_idx(4,1): -1})
        add_term({u_idx(0,0): 1}, {v_idx(0,4): 1, v_idx(1,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,3): 1, v_idx(0,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(3,0): -1, w_idx(4,0): 1})
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,3): 1, v_idx(0,4): 1}, {w_idx(4,1): 1})
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1})
        
        # Part 4: Strassen for A[0:2, 0:2] * B[0:2, 5:7]
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1, w_idx(6,1): 1})
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1}, {w_idx(5,1): 1, w_idx(6,1): -1})
        add_term({u_idx(0,0): 1}, {v_idx(0,6): 1, v_idx(1,6): -1}, {w_idx(6,0): 1, w_idx(6,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,5): 1, v_idx(0,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(5,0): -1, w_idx(6,0): 1})
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,5): 1, v_idx(0,6): 1}, {w_idx(6,1): 1})
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1})
        
        # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
        for k in range(p):
            add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
            add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
            
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for c, (u_list, v_list, w_list) in enumerate(terms):
            for i in range(n * m):
                U[i][c] = u_list[i]
            for i in range(m * p):
                V[i][c] = v_list[i]
            for i in range(p * n):
                W[i][c] = w_list[i]
        return {"U": U, "V": V, "W": W}

    # Search for lower rank
    # We have ~120s.
    # Try ranks 38 down to 30.
    # Allocate time based on rank.
    start_search_time = time.time()
    max_search_time = 110.0
    
    for R in range(38, 29, -1):
        if time.time() - start_search_time > max_search_time:
            break
        # Allocate remaining time proportionally or fixed per rank
        # Let's give 10s per rank
        result = try_rank(R, max_restarts=5, timeout=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no better rank found, return rank 39
    return get_rank_39()

# EVOLVE-BLOCK-END
