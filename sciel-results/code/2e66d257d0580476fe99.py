import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_solution(U, V, W):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_hat, T, atol=1e-9)

    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial))
    V_trivial = np.zeros((mp, rank_trivial))
    W_trivial = np.zeros((pn, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, r] = 1
                V_trivial[b, r] = 1
                W_trivial[c, r] = 1
                r += 1

    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = rank_trivial

    start_time = time.time()
    deadline = start_time + 110

    def get_residuals(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_hat - T).flatten()

    # Search for lower rank using ALS + Polish
    for R in range(15, 25):
        if time.time() > deadline:
            break
        
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(seed + R * 100)
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # ALS iterations
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update W
                for c in range(pn):
                    M = np.einsum('ar,br->ab r', U, V).reshape(nm * mp, R)
                    rhs = T[:, :, c].flatten()
                    W_c, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    W[c, :] = W_c

                # Update V
                for b in range(mp):
                    M = np.einsum('ar,cr->ac r', U, W).reshape(nm * pn, R)
                    rhs = T[:, b, :].flatten()
                    V_b, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    V[b, :] = V_b

                # Update U
                for a in range(nm):
                    M = np.einsum('br,cr->bc r', V, W).reshape(mp * pn, R)
                    rhs = T[a, :, :].flatten()
                    U_a, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    U[a, :] = U_a
                
                # Check residual
                T_hat = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(T_hat - T)
                
                if residual < 1e-6:
                    # Polish with least_squares
                    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    try:
                        result = least_squares(
                            get_residuals, 
                            params, 
                            args=(R,), 
                            method='lm', 
                            max_nfev=1000,
                            xtol=1e-12,
                            ftol=1e-12
                        )
                        if result.success or result.cost < 1e-12:
                            params = result.x
                            U = params[:R*nm].reshape(nm, R)
                            V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                            W = params[R*nm+R*mp:].reshape(pn, R)
                            
                            U_snap = np.round(U * 2) / 2
                            V_snap = np.round(V * 2) / 2
                            W_snap = np.round(W * 2) / 2
                            
                            if verify_solution(U_snap, V_snap, W_snap):
                                return {
                                    "U": U_snap.tolist(),
                                    "V": V_snap.tolist(),
                                    "W": W_snap.tolist()
                                }
                    except Exception:
                        pass
                    break

    return best_result
# EVOLVE-BLOCK-END
