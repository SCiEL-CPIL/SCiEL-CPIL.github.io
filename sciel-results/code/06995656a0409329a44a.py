import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Target tensor T
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
    
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Ranks to try
    # Schoolbook is 54.
    # Known lower bound for 3x3x3 is 23.
    # For 3x3x6, maybe 38?
    # Let's try 36, 37, 38.
    ranks_to_try = [36, 37, 38]
    
    best_result = None
    best_rank = 54
    
    def unpack(x):
        # x is flattened [U, V, W]
        # U: (nm, r), V: (mp, r), W: (np_, r)
        # We store them as (r, nm), (r, mp), (r, np_) in x for easier slicing?
        # Let's define x layout: 
        # First r*nm entries: U (r rows, nm cols)
        # Next r*mp entries: V (r rows, mp cols)
        # Next r*np_ entries: W (r rows, np_ cols)
        U = x[:r*nm].reshape(r, nm).T
        V = x[r*nm:r*nm + r*mp].reshape(r, mp).T
        W = x[r*nm + r*mp:r*nm + r*mp + r*np_].reshape(r, np_).T
        return U, V, W

    def residual(x):
        U, V, W = unpack(x)
        # Vectorized contraction
        # 'xc,yc,zc->xyz'
        # U is (nm, r) -> xc
        # V is (mp, r) -> yc
        # W is (np_, r) -> zc
        tensor = np.einsum('xc,yc,zc->xyz', U, V, W)
        return tensor.flatten() - T_flat

    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Number of restarts depends on time
        # Let's do 20 restarts for r=36, 20 for 37, 20 for 38
        # Or adaptively.
        max_restarts = 30
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale: entries in T are 1. 
            # If U, V, W are ~1, sum of r terms is ~r.
            # So we should scale down by 1/sqrt(r) or similar.
            x0 = np.random.randn(r * (nm + mp + np_)) * (1.0 / np.sqrt(r))
            
            try:
                from scipy.optimize import least_squares
                # Use 'lm' if possible, else 'trf'
                # 'lm' is faster but requires M >= N?
                # M = 2916, N = 12r.
                # For r=38, N=456. M > N. So 'lm' is applicable.
                res = least_squares(residual, x0, method='lm', max_nfev=5000, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                err = np.linalg.norm(res.fun)
                
                if err < 1e-5:
                    U, V, W = unpack(res.x)
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    tensor_check = np.einsum('xc,yc,zc->xyz', U_r, V_r, W_r)
                    if np.allclose(tensor_check.flatten(), T_flat, atol=1e-9):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                    # If rounding fails, maybe try integer rounding?
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    tensor_check_i = np.einsum('xc,yc,zc->xyz', U_i, V_i, W_i)
                    if np.allclose(tensor_check_i.flatten(), T_flat, atol=1e-9):
                        return {
                            "U": U_i.tolist(),
                            "V": V_i.tolist(),
                            "W": W_i.tolist()
                        }
                        
            except Exception:
                pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
