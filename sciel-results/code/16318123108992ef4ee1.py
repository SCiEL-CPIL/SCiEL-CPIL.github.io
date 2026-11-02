import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    # The tensor T_{i,j,k} = delta_{j,k}
    # We want to find U (9 x r), V (9 x r), W (9 x r) such that
    # T_{i,j,k} = sum_{a=1}^r U_{i,j,a} V_{j,k,a} W_{k,i,a}
    # Note: the indices in the task are:
    # U[i*m+j, r], V[l*p+k, r], W[k*n+i, r]
    # So the term is U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
    # And we want sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = delta_{j,k}

    # Schoolbook rank 27
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
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = rank_sb

    def verify(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    expected = 1.0 if j == k else 0.0
                    s = 0.0
                    for a in range(rank):
                        s += U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
                    if abs(s - expected) > tol:
                        return False
        return True

    def objective(vars, target_rank):
        r = target_rank
        U_f = vars[:nm*r].reshape(nm, r)
        V_f = vars[nm*r:2*nm*r].reshape(mp, r)
        W_f = vars[2*nm*r:3*nm*r].reshape(pn, r)
        
        errors = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    expected = 1.0 if j == k else 0.0
                    s = np.sum(W_f[k*n+i, :] * U_f[i*m+j, :] * V_f[j*p+k, :])
                    errors.append(s - expected)
        return np.array(errors)

    def try_reduce_rank(target_rank, num_starts=5):
        for start_idx in range(num_starts):
            if time.time() > deadline - 5:
                return None
            
            if start_idx == 0:
                np.random.seed(42 + start_idx)
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                keep_indices = np.random.choice(rank_sb, target_rank, replace=False)
                sort_keep = np.sort(keep_indices)
                
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                for new_col, old_col in enumerate(sort_keep):
                    U_init[:, new_col] = U[:, old_col]
                    V_init[:, new_col] = V[:, old_col]
                    W_init[:, new_col] = W[:, old_col]
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=2000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-6):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
        return None

    # Try rank 23
    if time.time() - start_time < deadline - 10:
        result = try_reduce_rank(23, num_starts=10)
        if result:
            best_U, best_V, best_W = result
            best_rank = 23
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 26
    if time.time() - start_time < deadline - 10:
        result = try_reduce_rank(26, num_starts=10)
        if result:
            best_U, best_V, best_W = result
            best_rank = 26
            return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
