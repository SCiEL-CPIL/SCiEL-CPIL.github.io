"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import scipy.optimize as so

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    U_dims = n * m  # 6
    V_dims = m * p  # 12
    W_dims = p * n  # 8
    
    T_dense = np.zeros((U_dims, V_dims, W_dims))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_dense[a, b, c] = 1.0

    def smooth_cp(rank, seed=0, max_iter=500):
        """
        Use scipy.optimize.minimize to find a CP decomposition.
        Parameters: U (U_dims x rank), V (V_dims x rank), W (W_dims x rank)
        Objective: || T - sum U_t V_t W_t ||_F^2
        """
        rng = random.Random(seed)
        
        # Initialize with small random values
        U = rng.gauss(0, 1, (U_dims, rank)) * 0.5
        V = rng.gauss(0, 1, (V_dims, rank)) * 0.5
        W = rng.gauss(0, 1, (W_dims, rank)) * 0.5
        
        # Flatten for optimizer
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_x = x[:U_dims*rank].reshape(U_dims, rank)
            V_x = x[U_dims*rank:U_dims*rank + V_dims*rank].reshape(V_dims, rank)
            W_x = x[U_dims*rank + V_dims*rank:].reshape(W_dims, rank)
            
            T_rec = np.einsum('at,bt,ct->abc', U_x, V_x, W_x)
            diff = T_rec - T_dense
            return np.sum(diff * diff)
        
        def gradient(x):
            U_x = x[:U_dims*rank].reshape(U_dims, rank)
            V_x = x[U_dims*rank:U_dims*rank + V_dims*rank].reshape(V_dims, rank)
            W_x = x[U_dims*rank + V_dims*rank:].reshape(W_dims, rank)
            
            T_rec = np.einsum('at,bt,ct->abc', U_x, V_x, W_x)
            diff = T_rec - T_dense
            
            gU = np.einsum('abc,bt,ct->at', diff, V_x, W_x)
            gV = np.einsum('abc,at,ct->bt', diff, U_x, W_x)
            gW = np.einsum('abc,at,bt->ct', diff, U_x, V_x)
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

        try:
            res = so.minimize(
                objective, x0, jac=gradient, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            return None, float('inf')
        
        x_opt = res.x
        U_opt = x_opt[:U_dims*rank].reshape(U_dims, rank)
        V_opt = x_opt[U_dims*rank:U_dims*rank + V_dims*rank].reshape(V_dims, rank)
        W_opt = x_opt[U_dims*rank + V_dims*rank:].reshape(W_dims, rank)
        
        err = np.sqrt(res.fun)
        return (U_opt, V_opt, W_opt), err

    def snap_to_half_integers(U, V, W, tol=1e-6):
        """Snap to half-integers and verify"""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        T_rec = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
        err = np.linalg.norm(T_rec - T_dense)
        if err < tol:
            return U_snap, V_snap, W_snap
        return None

    def to_list(mat):
        result = []
        for row in mat:
            rrow = []
            for val in row:
                v2 = round(val * 2)
                if abs(val * 2 - v2) < 1e-9:
                    rrow.append(v2 / 2.0)
                else:
                    rrow.append(float(val))
            result.append(rrow)
        return result

    best_result = None
    # Try ranks from 19 to 24
    for r in range(19, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        # L-BFGS-B is fast, allow more restarts
        max_restarts = 200 if r < 22 else 100
        
        for seed in range(max_restarts):
            if time.time() > deadline:
                break
            try:
                res_tuple, err = smooth_cp(r, seed=seed * 7919 + r, max_iter=100)
                if res_tuple is not None:
                    U, V, W = res_tuple
                    if err < 1e-8:
                        snapped = snap_to_half_integers(U, V, W)
                        if snapped is not None:
                            U_s, V_s, W_s = snapped
                            best_result = {
                                'U': to_list(U_s),
                                'V': to_list(V_s),
                                'W': to_list(W_s)
                            }
                            break
            except Exception:
                continue
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
