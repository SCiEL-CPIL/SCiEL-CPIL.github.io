"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor():
    n, m, p = 2, 3, 6
    T = np.zeros((p*n, n*m, m*p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = k * n + i
                a_idx = i * m + j
                b_idx = j * p + k
                T[out_idx, a_idx, b_idx] = 1.0
    return T

def _als_decompose(T, r, max_iter=200, tol=1e-10, seed=0):
    rng = np.random.RandomState(seed)
    n_out, n_A, n_B = T.shape
    task_W = rng.randn(n_out, r)
    task_U = rng.randn(n_A, r)
    task_V = rng.randn(n_B, r)
    
    for it in range(max_iter):
        UV = np.einsum('ac,bc->abc', task_U, task_V)
        UV2 = UV.reshape(n_A * n_B, r)
        T0 = T.reshape(n_out, n_A * n_B)
        A_mat = UV2.T @ UV2
        b_mat = T0.T @ UV2
        try:
            task_W = np.linalg.solve(A_mat, b_mat).T
        except:
            task_W = np.linalg.lstsq(A_mat, b_mat, rcond=None)[0].T
        
        WV = np.einsum('oc,bc->obc', task_W, task_V)
        WV2 = WV.reshape(n_out * n_B, r)
        T1 = T.transpose(1, 0, 2).reshape(n_A, n_out * n_B)
        A_mat = WV2.T @ WV2
        b_mat = T1.T @ WV2
        try:
            task_U = np.linalg.solve(A_mat, b_mat).T
        except:
            task_U = np.linalg.lstsq(A_mat, b_mat, rcond=None)[0].T
        
        WU = np.einsum('oc,ac->oac', task_W, task_U)
        WU2 = WU.reshape(n_out * n_A, r)
        T2 = T.transpose(2, 0, 1).reshape(n_B, n_out * n_A)
        A_mat = WU2.T @ WU2
        b_mat = T2.T @ WU2
        try:
            task_V = np.linalg.solve(A_mat, b_mat).T
        except:
            task_V = np.linalg.lstsq(A_mat, b_mat, rcond=None)[0].T
        
        recon = np.einsum('oc,ac,bc->oab', task_W, task_U, task_V)
        err = np.linalg.norm(recon - T) / np.linalg.norm(T)
        if err < tol:
            return task_U, task_V, task_W, err
    
    recon = np.einsum('oc,ac,bc->oab', task_W, task_U, task_V)
    err = np.linalg.norm(recon - T) / np.linalg.norm(T)
    if err < 1e-6:
        return task_U, task_V, task_W, err
    return None

def _round_to_half(x):
    return np.round(2 * x) / 2.0

def _verify(T, U, V, W):
    recon = np.einsum('ac,bc,oc->oab', U, V, W)
    return np.allclose(recon, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    T = _build_tensor()
    n_out, n_A, n_B = T.shape
    
    best_result = None
    best_rank = None
    
    lower_bound = max(n_out, n_A, n_B)
    for r in range(lower_bound, lower_bound + 15):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            result = _als_decompose(T, r, max_iter=150, seed=seed)
            if result is not None:
                task_U, task_V, task_W, err = result
                # Round to half-integers
                U_r = _round_to_half(task_U)
                V_r = _round_to_half(task_V)
                W_r = _round_to_half(task_W)
                if _verify(T, U_r, V_r, W_r):
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    best_rank = r
                    found = True
                    break
                # If rounding doesn't work, try without rounding (but task requires half-integers)
                # For now, skip
        if found:
            break
    
    if best_result is None:
        # Fallback to schoolbook (all 1s, which are integers)
        n, m, p = 2, 3, 6
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
