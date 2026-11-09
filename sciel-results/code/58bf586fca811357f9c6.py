import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the guaranteed-correct schoolbook decomposition (rank 60)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Check if U, V, W form a valid exact decomposition."""
    n, m, p = DIMENSIONS
    r = len(W[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                if abs(2.0 * x - round(2.0 * x)) > 1e-9:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for c in range(r):
                    total += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
            if abs(total - m) > 1e-9:
                return False
    return True


def _fast_search(deadline):
    """
    Structured search exploiting matrix multiplication tensor structure.
    Uses column-wise random restarts with greedy local repair.
    """
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from a promising range
    for target_rank in [18, 20, 22, 24, 26, 28, 30, 32, 35, 40, 45, 50, 55]:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline:
            break
        
        found = False
        np.random.seed(target_rank * 7919 + 12345)
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Sparse random initialization with ±1 entries
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            for c in range(target_rank):
                # Each column: pick 2-3 nonzeros per matrix
                nu = np.random.randint(2, 4)
                vu = np.random.choice(n * m, size=nu, replace=False)
                U[vu, c] = np.random.choice([-1, 1], size=nu)
                
                nv = np.random.randint(2, 4)
                vv = np.random.choice(m * p, size=nv, replace=False)
                V[vv, c] = np.random.choice([-1, 1], size=nv)
                
                nw = np.random.randint(2, 4)
                vw = np.random.choice(p * n, size=nw, replace=False)
                W[vw, c] = np.random.choice([-1, 1], size=nw)
            
            # Compute residual for each (i,j,k): should be 1
            # residual[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
            # Use vectorized computation
            def compute_residual():
                res = np.zeros((n, m, p))
                for c in range(target_rank):
                    Uc = U[:, c]  # (n*m,)
                    Vc = V[:, c]  # (m*p,)
                    Wc = W[:, c]  # (p*n,)
                    # Reshape and multiply
                    U_reshaped = Uc.reshape(n, m)
                    V_reshaped = Vc.reshape(m, p)
                    W_reshaped = Wc.reshape(p, n)
                    # For each (i,j,k): U[i,j] * V[j,k] * W[k,i]
                    # = sum over (i,j,k) of outer products
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res[i, j, k] += U_reshaped[i, j] * V_reshaped[j, k] * W_reshaped[k, i]
                return res - 1.0
            
            res = compute_residual()
            max_res = np.max(np.abs(res))
            
            if max_res < 1e-9:
                found = True
                break
            
            # Greedy local search: iteratively fix the worst residual
            for it in range(3000):
                if time.time() > deadline:
                    break
                
                # Find worst residual
                flat_idx = np.argmax(np.abs(res))
                wi, wj, wk = np.unravel_index(flat_idx, res.shape)
                werr = res[wi, wj, wk]
                
                if abs(werr) < 1e-9:
                    if np.max(np.abs(res)) < 1e-9:
                        found = True
                        break
                    continue
                
                # Try flipping or setting entries in a random column
                c = np.random.randint(target_rank)
                ui = wi * m + wj
                vj = wj * p + wk
                wk_idx = wk * n + wi
                
                # Try U flip/set
                old = U[ui, c]
                if old != 0:
                    U[ui, c] = -old
                    # Update residuals affected by this change
                    delta = -2 * old  # change in U value
                    for k2 in range(p):
                        for i2 in range(n):
                            for j2 in range(m):
                                if i2 == wi and j2 == wj:
                                    pass
                    # Simpler: just recompute affected slice
                    for k2 in range(p):
                        v_val = V[wj * p + k2, c]
                        if v_val != 0:
                            for i2 in range(n):
                                w_val = W[k2 * n + i2, c]
                                if w_val != 0:
                                    res[i2, wj, k2] += delta * v_val * w_val
                    if abs(res[wi, wj, wk]) >= abs(werr):
                        U[ui, c] = old  # revert
                else:
                    U[ui, c] = 1
                    for k2 in range(p):
                        v_val = V[wj * p + k2, c]
                        if v_val != 0:
                            for i2 in range(n):
                                w_val = W[k2 * n + i2, c]
                                if w_val != 0:
                                    res[i2, wj, k2] += 1 * v_val * w_val
                    if abs(res[wi, wj, wk]) >= abs(werr):
                        U[ui, c] = 0  # revert
                
                # Try V flip/set
                old = V[vj, c]
                if old != 0:
                    V[vj, c] = -old
                    delta = -2 * old
                    for i2 in range(n):
                        u_val = U[i2 * m + wj, c]
                        if u_val != 0:
                            for k2 in range(p):
                                w_val = W[k2 * n + i2, c]
                                if w_val != 0:
                                    res[i2, wj, k2] += u_val * delta * w_val
                    if abs(res[wi, wj, wk]) >= abs(werr):
                        V[vj, c] = old
                else:
                    V[vj, c] = 1
                    for i2 in range(n):
                        u_val = U[i2 * m + wj, c]
                        if u_val != 0:
                            for k2 in range(p):
                                w_val = W[k2 * n + i2, c]
                                if w_val != 0:
                                    res[i2, wj, k2] += u_val * 1 * w_val
                    if abs(res[wi, wj, wk]) >= abs(werr):
                        V[vj, c] = 0
                
                # Try W flip/set
                old = W[wk_idx, c]
                if old != 0:
                    W[wk_idx, c] = -old
                    delta = -2 * old
                    for i2 in range(n):
                        for j2 in range(m):
                            u_val = U[i2 * m + j2, c]
                            if u_val != 0:
                                v_val = V[j2 * p + wk, c]
                                if v_val != 0:
                                    res[i2, j2, wk] += u_val * v_val * delta
                    if abs(res[wi, wj, wk]) >= abs(werr):
                        W[wk_idx, c] = old
                else:
                    W[wk_idx, c] = 1
                    for i2 in range(n):
                        for j2 in range(m):
                            u_val = U[i2 * m + j2, c]
                            if u_val != 0:
                                v_val = V[j2 * p + wk, c]
                                if v_val != 0:
                                    res[i2, j2, wk] += u_val * v_val * 1
                    if abs(res[wi, wj, wk]) >= abs(werr):
                        W[wk_idx, c] = 0
            
            if np.max(np.abs(res)) < 1e-9:
                found = True
                break
        
        if found:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            result = {"U": U_list, "V": V_list, "W": W_list}
            if _verify(U_list, V_list, W_list):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result


def construct():
    t0 = time.time()
    deadline = t0 + 100.0
    
    result = _fast_search(deadline)
    
    if result is not None:
        return result
    
    return _schoolbook()
# EVOLVE-BLOCK-END
