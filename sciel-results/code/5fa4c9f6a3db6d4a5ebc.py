"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build full set of constraints: all entries of the tensor
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    # T[u, v, w] = 0 otherwise
    constraints = []
    # Set of non-zero (u, v, w)
    non_zero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zero_set.add((u, v, w))
    
    # Iterate over all possible (u, v, w)
    for u in range(nm):
        for v in range(mp):
            for w in range(np_):
                if (u, v, w) in non_zero_set:
                    constraints.append((u, v, w, 1.0))
                else:
                    constraints.append((u, v, w, 0.0))
    
    n_eq = len(constraints)
    
    def residuals(params, rank):
        U = params[:nm * rank].reshape(nm, rank)
        V = params[nm * rank:(nm + mp) * rank].reshape(mp, rank)
        W = params[(nm + mp) * rank:].reshape(np_, rank)
        
        res = np.zeros(n_eq)
        for idx, (u, v, w, val) in enumerate(constraints):
            res[idx] = np.dot(U[u], V[v] * W[w]) - val
        return res
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Known minimal rank for 2x3x4 is 13
    for rank in range(13, 25):
        if time.time() > deadline:
            break
            
        for restart in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + rank * 100 + restart)
            U_init = np.random.randn(nm, rank) * 0.5
            V_init = np.random.randn(mp, rank) * 0.5
            W_init = np.random.randn(np_, rank) * 0.5
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, params_init, args=(rank,), 
                                      method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if result.cost < 1e-10:
                U = result.x[:nm * rank].reshape(nm, rank)
                V = result.x[nm * rank:(nm + mp) * rank].reshape(mp, rank)
                W = result.x[(nm + mp) * rank:].reshape(np_, rank)
                
                # Verify full tensor exactly
                full_ok = True
                for u in range(nm):
                    for v in range(mp):
                        for w in range(np_):
                            expected = 1.0 if (u, v, w) in non_zero_set else 0.0
                            actual = np.dot(U[u], V[v] * W[w])
                            if abs(actual - expected) > 1e-8:
                                full_ok = False
                                break
                        if not full_ok:
                            break
                    if not full_ok:
                        break
                if not full_ok:
                    continue
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
