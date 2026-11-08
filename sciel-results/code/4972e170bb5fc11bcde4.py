import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_start_time = time.time()
TIME_LIMIT = 140.0

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification."""
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Compute T_decomp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T_decomp = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T_decomp, T_target, atol=1e-9)

def _rank50_baseline():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    result = _rank50_baseline()
    
    if time.time() - _start_time > TIME_LIMIT - 10:
        return result

    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Systematic column removal
    for _ in range(best_rank):
        if time.time() - _start_time > TIME_LIMIT - 5:
            break
        removed_any = False
        for r in range(best_U.shape[1]):
            if time.time() - _start_time > TIME_LIMIT - 5:
                break
            U_test = np.delete(best_U, r, axis=1)
            V_test = np.delete(best_V, r, axis=1)
            W_test = np.delete(best_W, r, axis=1)
            
            if _verify_fast(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                removed_any = True
                break
        if not removed_any:
            break
    
    # Heuristic search to reduce rank further
    for _ in range(5000):
        if time.time() - _start_time > TIME_LIMIT - 5:
            break
            
        # Randomly perturb
        U_pert = best_U.copy()
        V_pert = best_V.copy()
        W_pert = best_W.copy()
        
        for _ in range(3):
            mat = [U_pert, V_pert, W_pert][np.random.randint(3)]
            i, j = np.random.randint(mat.shape[0]), np.random.randint(mat.shape[1])
            mat[i, j] += np.random.choice([-1, 1]) * 0.5
            
        # Round to nearest half-integer
        U_round = np.round(U_pert * 2) / 2
        V_round = np.round(V_pert * 2) / 2
        W_round = np.round(W_pert * 2) / 2
        
        # Remove zero columns
        new_rank = U_round.shape[1]
        for r in range(new_rank):
            if np.allclose(U_round[:, r], 0) and np.allclose(V_round[:, r], 0) and np.allclose(W_round[:, r], 0):
                U_round = np.delete(U_round, r, axis=1)
                V_round = np.delete(V_round, r, axis=1)
                W_round = np.delete(W_round, r, axis=1)
                new_rank -= 1
                r -= 1
        
        if new_rank < best_rank:
            if _verify_fast(U_round, V_round, W_round, n, m, p):
                best_rank = new_rank
                best_U, best_V, best_W = U_round.copy(), V_round.copy(), W_round.copy()
                if best_rank <= 40:
                    break

    if best_rank < 50:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return result
