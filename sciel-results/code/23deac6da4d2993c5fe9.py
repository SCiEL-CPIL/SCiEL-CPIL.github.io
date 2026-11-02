import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

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

    def objective(vars, target_rank):
        r = target_rank
        U_f = vars[:nm*r].reshape(nm, r)
        V_f = vars[nm*r:2*nm*r].reshape(mp, r)
        W_f = vars[2*nm*r:3*nm*r].reshape(pn, r)
        
        errors = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W_f[k*n+i, :] * U_f[i*m+j, :] * V_f[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        errors.append(s - expected)
        return np.array(errors)

    def try_reduce_rank(target_rank, num_starts=10):
        # Strategy: Start from schoolbook, zero out (27 - target_rank) columns, then optimize.
        # We need to choose which columns to keep.
        # Keeping all is rank 27.
        # We will try different subsets of columns to keep.
        
        # For simplicity, let's just try to optimize from a random start first, 
        # and if that fails, try the reduction strategy.
        
        # Reduction Strategy:
        # We have 27 columns. We want to keep `target_rank` of them.
        # We will iterate through combinations of columns to drop? 27 choose 4 is 17550. Too many.
        # Instead, we can just pick random sets of columns to drop.
        
        for start_idx in range(num_starts):
            if time.time() > deadline - 5:
                return None
            
            if start_idx == 0:
                # Try random initialization
                np.random.seed(42 + start_idx)
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Try reduction from schoolbook
                # Pick random indices to keep
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
                res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                
                # Round to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-3):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # If rounding failed, try to polish the rounded values
                # This is complex, so we just try another seed.
                
        return None

    # Try rank 23
    if time.time() - start_time < deadline - 10:
        result = try_reduce_rank(23, num_starts=20)
        if result:
            best_U, best_V, best_W = result
            best_rank = 23
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 26
    if time.time() - start_time < deadline - 10:
        result = try_reduce_rank(26, num_starts=20)
        if result:
            best_U, best_V, best_W = result
            best_rank = 26
            return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
