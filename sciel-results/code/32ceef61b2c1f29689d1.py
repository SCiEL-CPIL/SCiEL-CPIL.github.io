import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    # Precompute the target tensor indices
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_indices.append((a, b, c))
    
    ones_set = set(ones_indices)
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in ones_set:
                        continue
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s) > tol:
                        return False
        return True
    
    # Rank-26 Strassen-based decomposition for 3x3 matrix multiplication
    # Uses 7 Strassen terms for the 2x2 core plus 19 border terms.
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen terms for the 2x2 core (rows/cols 0,1)
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    add_term({3: 1, 4: 1}, {0: 1}, {3: 1, 4: 1})
    add_term({0: 1}, {4: 1, 0: -1}, {0: -1, 4: 1})
    add_term({4: 1}, {3: 1, 0: -1}, {3: -1, 4: 1})
    add_term({0: 1, 1: 1}, {4: 1}, {1: 1, 4: 1})
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1, 1: -1})
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {4: 1, 1: -1})
    
    # Border terms
    add_term({0: 1}, {2: 1}, {2: 1})
    add_term({1: 1}, {5: 1}, {2: 1})
    add_term({2: 1}, {8: 1}, {2: 1})
    
    add_term({6: 1}, {0: 1}, {6: 1})
    add_term({7: 1}, {3: 1}, {6: 1})
    add_term({8: 1}, {6: 1}, {6: 1})
    
    add_term({6: 1}, {2: 1}, {8: 1})
    add_term({7: 1}, {5: 1}, {8: 1})
    add_term({8: 1}, {8: 1}, {8: 1})
    
    add_term({3: 1}, {2: 1}, {5: 1})
    add_term({4: 1}, {5: 1}, {5: 1})
    add_term({5: 1}, {8: 1}, {5: 1})
    
    add_term({6: 1}, {1: 1}, {7: 1})
    add_term({7: 1}, {4: 1}, {7: 1})
    add_term({8: 1}, {7: 1}, {7: 1})
    
    # Build the rank-22 candidate and verify
    r22 = len(terms)
    U22 = np.zeros((nm, r22))
    V22 = np.zeros((mp, r22))
    W22 = np.zeros((pn, r22))
    for col_idx, (u_d, v_d, w_d) in enumerate(terms):
        for idx, val in u_d.items():
            U22[idx, col_idx] = val
        for idx, val in v_d.items():
            V22[idx, col_idx] = val
        for idx, val in w_d.items():
            W22[idx, col_idx] = val
    
    # Check if rank-22 works
    if verify(U22, V22, W22, tol=1e-9):
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
    
    # Fallback: schoolbook rank 27
    rank_sb = 27
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    # Try rank 26 with optimizer
    from scipy.optimize import least_squares
    
    all_indices = [(a,b,c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]
    
    for r in [26, 25, 24]:
        if time.time() - start_time > deadline - 20:
            break
            
        for attempt in range(15):
            if time.time() - start_time > deadline - 5:
                break
                
            np.random.seed(attempt * 7919 + r)
            
            # Warm start from schoolbook (take first r terms)
            U_init = U[:, :r].copy()
            V_init = V[:, :r].copy()
            W_init = W[:, :r].copy()
            # Add perturbation
            U_init += np.random.randn(nm, r) * 0.3
            V_init += np.random.randn(mp, r) * 0.3
            W_init += np.random.randn(pn, r) * 0.3
            
            shuffled_zeros = zero_indices.copy()
            np.random.shuffle(shuffled_zeros)
            opt_zero_indices = shuffled_zeros[:200]
            
            def residual_full(x, r=r, ones_indices=ones_indices, opt_zero_indices=opt_zero_indices, nm=nm, mp=mp, pn=pn):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:2*nm*r].reshape(mp, r)
                W_f = x[2*nm*r:3*nm*r].reshape(pn, r)
                
                res = []
                for a, b, c in ones_indices:
                    s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                    res.append(s - 1.0)
                for a, b, c in opt_zero_indices:
                    s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                    res.append(s)
                return np.array(res)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual_full, x0, method='trf', max_nfev=3000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                
                if sol.cost < 1e-10:
                    U_sol = sol.x[:nm*r].reshape(nm, r)
                    V_sol = sol.x[nm*r:2*nm*r].reshape(mp, r)
                    W_sol = sol.x[2*nm*r:3*nm*r].reshape(pn, r)
                    
                    # Try rounding to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if verify(U_r, V_r, W_r, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        break
            except Exception:
                pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
