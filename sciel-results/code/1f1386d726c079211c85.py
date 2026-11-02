import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0

def _verify_decomposition(U, V, W):
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_36():
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1

    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _round_to_half_int(X):
    return np.round(2 * X) / 2.0

def _alternating_minimize(U, V, W, n_iter=200, tol=1e-10):
    n, m, p = 2, 4, 5
    r = U.shape[1]
    
    for it in range(n_iter):
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    pass
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        if max_err < tol:
            break
    
    return U, V, W, max_err

def _try_reduce_rank(U_start, V_start, W_start, target_rank, max_trials=50):
    n, m, p = 2, 4, 5
    rank_start = U_start.shape[1]
    cols_to_remove = rank_start - target_rank
    if cols_to_remove < 0:
        return None
        
    rng = np.random.RandomState(42)
    
    # Try systematic removal first if small
    if cols_to_remove <= 3:
        for combo in combinations(range(rank_start), cols_to_remove):
            mask = [i for i in range(rank_start) if i not in combo]
            U_curr = U_start[:, mask].copy()
            V_curr = V_start[:, mask].copy()
            W_curr = W_start[:, mask].copy()
            U_sol, V_sol, W_sol, err = _alternating_minimize(U_curr, V_curr, W_curr, n_iter=200, tol=1e-10)
            if err < 1e-8:
                U_r = _round_to_half_int(U_sol)
                V_r = _round_to_half_int(V_sol)
                W_r = _round_to_half_int(W_sol)
                if _verify_decomposition(U_r, V_r, W_r):
                    return U_r, V_r, W_r
    else:
        # Random removal
        for trial in range(max_trials):
            indices_to_remove = rng.choice(rank_start, size=cols_to_remove, replace=False)
            mask = [i for i in range(rank_start) if i not in indices_to_remove]
            U_curr = U_start[:, mask].copy()
            V_curr = V_start[:, mask].copy()
            W_curr = W_start[:, mask].copy()
            
            # Add small noise to break symmetry
            noise = 0.01 * rng.randn(*U_curr.shape)
            U_curr += noise
            V_curr += 0.01 * rng.randn(*V_curr.shape)
            W_curr += 0.01 * rng.randn(*W_curr.shape)
            
            U_sol, V_sol, W_sol, err = _alternating_minimize(U_curr, V_curr, W_curr, n_iter=300, tol=1e-10)
            if err < 1e-8:
                U_r = _round_to_half_int(U_sol)
                V_r = _round_to_half_int(V_sol)
                W_r = _round_to_half_int(W_sol)
                if _verify_decomposition(U_r, V_r, W_r):
                    return U_r, V_r, W_r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook 40
        n, m, p = 2, 4, 5
        rank = 40
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 36
    
    # Try to reduce to 32
    if time.time() < deadline - 20:
        result = _try_reduce_rank(U, V, W, 32, max_trials=100)
        if result:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = 32
            
    # If 32 failed, try 33, 34, 35
    for target in [35, 34, 33]:
        if best_rank <= target:
            continue
        if time.time() < deadline - 10:
            result = _try_reduce_rank(U, V, W, target, max_trials=50)
            if result:
                U, V, W = result
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target
                if best_rank == 32:
                    break

    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
