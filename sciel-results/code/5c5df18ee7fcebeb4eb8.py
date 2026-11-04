import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]
_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE:
        # Return a trivial rank-64 (schoolbook) decomposition as fallback
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Default rank 56 solution (Strassen based)
    rank_56 = 56
    U_56 = [[0] * rank_56 for _ in range(n * m)]
    V_56 = [[0] * rank_56 for _ in range(m * p)]
    W_56 = [[0] * rank_56 for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block_56(base_rank, i_base, j_base, k_base, U, V, W):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val

    # Populate rank 56
    add_block_56(0, 0, 0, 0, U_56, V_56, W_56)
    add_block_56(7, 0, 0, 2, U_56, V_56, W_56)
    add_block_56(14, 0, 0, 4, U_56, V_56, W_56)
    add_block_56(21, 0, 0, 6, U_56, V_56, W_56)
    add_block_56(28, 0, 2, 0, U_56, V_56, W_56)
    add_block_56(35, 0, 2, 2, U_56, V_56, W_56)
    add_block_56(42, 0, 2, 4, U_56, V_56, W_56)
    add_block_56(49, 0, 2, 6, U_56, V_56, W_56)

    # Try to find a lower rank decomposition for 2x2x4
    # If found, use it to build a lower rank for 2x4x8
    # Target ranks: 10, 11, 12, 13
    target_ranks = [10, 11, 12, 13]
    n_sub, m_sub, p_sub = 2, 2, 4
    
    def objective_224(params, target_rank):
        U_flat = params[:4*target_rank]
        V_flat = params[4*target_rank:8*target_rank]
        W_flat = params[8*target_rank:]
        
        U = U_flat.reshape(2, 2, target_rank)
        V = V_flat.reshape(2, 2, target_rank)
        W = W_flat.reshape(4, 2, target_rank) # W is (k, i, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        target = np.ones((2, 2, 4))
        err = np.sum((T - target)**2)
        return err

    def grad_224(params, target_rank):
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = objective_224(params, target_rank)
        for i in range(len(params)):
            old = params[i]
            params[i] = old + eps
            f1 = objective_224(params, target_rank)
            params[i] = old
            grad[i] = (f1 - f0) / eps
        return grad

    found_params = None
    found_rank = None
    
    # Time budget for search: 100s
    search_deadline = time.time() + 100
    np.random.seed(42)
    
    for target_rank in target_ranks:
        if time.time() > search_deadline:
            break
            
        # Try to find a decomposition of this rank
        for restart in range(30):
            if time.time() > search_deadline:
                break
                
            # Random initialization
            params = np.random.randn(4*target_rank + 4*target_rank + 8*target_rank) * 0.1
            
            try:
                res = minimize(objective_224, params, args=(target_rank,), method='L-BFGS-B', 
                               jac=lambda x: grad_224(x, target_rank),
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
                if res.success and res.fun < 1e-8:
                    found_params = res.x
                    found_rank = target_rank
                    break
            except Exception:
                pass
        
        if found_rank is not None:
            break

    if found_rank is not None:
        # Construct the full decomposition
        rank_full = 4 * found_rank
        U_full = [[0] * rank_full for _ in range(n * m)]
        V_full = [[0] * rank_full for _ in range(m * p)]
        W_full = [[0] * rank_full for _ in range(p * n)]
        
        # Extract 2x2x4 decomposition
        U_224 = found_params[:4*found_rank].reshape(2, 2, found_rank)
        V_224 = found_params[4*found_rank:8*found_rank].reshape(2, 2, found_rank)
        W_224 = found_params[8*found_rank:].reshape(4, 2, found_rank)
        
        def add_block_224(base_rank, i_base, j_base, k_base, U, V, W):
            for r in range(found_rank):
                cr = base_rank + r
                for li in range(2):
                    for lj in range(2):
                        gi = i_base + li
                        gj = j_base + lj
                        U[gi * 4 + gj][cr] += U_224[li, lj, r]
                for lj in range(2):
                    for lk in range(4):
                        gj = j_base + lj
                        gk = k_base + lk
                        V[gj * 8 + gk][cr] += V_224[lj, lk, r]
                for lk in range(4):
                    for li in range(2):
                        gk = k_base + lk
                        gi = i_base + li
                        W[gk * 2 + gi][cr] += W_224[lk, li, r]
        
        # 4 blocks:
        # A0 (i=0,1; j=0,1) * B0a (j=0,1; k=0,1,2,3)
        add_block_224(0, 0, 0, 0, U_full, V_full, W_full)
        # A0 (i=0,1; j=0,1) * B0b (j=0,1; k=4,5,6,7)
        add_block_224(found_rank, 0, 0, 4, U_full, V_full, W_full)
        # A1 (i=0,1; j=2,3) * B1a (j=2,3; k=0,1,2,3)
        add_block_224(2*found_rank, 0, 2, 0, U_full, V_full, W_full)
        # A1 (i=0,1; j=2,3) * B1b (j=2,3; k=4,5,6,7)
        add_block_224(3*found_rank, 0, 2, 4, U_full, V_full, W_full)
        
        # Verify
        try:
            U_arr = np.array(U_full, dtype=float)
            V_arr = np.array(V_full, dtype=float)
            W_arr = np.array(W_full, dtype=float)
            U3 = U_arr.reshape(n, m, rank_full)
            V3 = V_arr.reshape(m, p, rank_full)
            W3 = W_arr.reshape(p, n, rank_full)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return {"U": U_full, "V": V_full, "W": W_full}
        except Exception:
            pass

    # Fallback to rank 56
    return {"U": U_56, "V": V_56, "W": W_56}
# EVOLVE-BLOCK-END
