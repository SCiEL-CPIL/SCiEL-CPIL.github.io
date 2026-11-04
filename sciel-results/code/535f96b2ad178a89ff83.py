import time
import numpy as np

DIMENSIONS = [2, 3, 9]
DIMENSIONS_233 = [2, 3, 3]

def _schoolbook():
    """Return schoolbook decomposition with rank 54."""
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

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly matches the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_233(U, V, W, tol=1e-8):
    """Verify decomposition for 2x3x3."""
    n, m, p = DIMENSIONS_233
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_233(rank, max_iters=2000, tol=1e-7, seed=42):
    """
    Try to find a decomposition of 2x3x3 of given rank using vectorized gradients.
    """
    n, m, p = DIMENSIONS_233
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(seed)
    
    # Precompute index arrays for vectorized operations
    idx_u_list, idx_v_list, idx_w_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u_list.append(i * m + j)
                idx_v_list.append(j * p + k)
                idx_w_list.append(k * n + i)
    idx_u_arr = np.array(idx_u_list)
    idx_v_arr = np.array(idx_v_list)
    idx_w_arr = np.array(idx_w_list)
    num_terms = len(idx_u_arr)
    
    def _optimize(U, V, W, lr_init=0.02, max_iters=max_iters, tol=tol):
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = lr_init
        momentum = 0.9
        
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        
        for it in range(max_iters):
            # Vectorized: compute all products at once
            U_sub = U[idx_u_arr, :]  # (num_terms, rank)
            V_sub = V[idx_v_arr, :]
            W_sub = W[idx_w_arr, :]
            
            vals = np.sum(U_sub * V_sub * W_sub, axis=1)  # (num_terms,)
            errs = vals - 1.0
            loss = np.mean(errs * errs)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < tol:
                break
            
            # Vectorized gradients
            # grad_U[i, :] = sum over terms where idx_u == i of 2*err*V*W
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # For each unique index, accumulate contributions
            for i in range(nm):
                mask = (idx_u_arr == i)
                if np.any(mask):
                    grad_U[i, :] = 2.0 * np.sum(errs[mask][:, None] * V_sub[mask] * W_sub[mask], axis=0)
            
            for j in range(mp):
                mask = (idx_v_arr == j)
                if np.any(mask):
                    grad_V[j, :] = 2.0 * np.sum(errs[mask][:, None] * U_sub[mask] * W_sub[mask], axis=0)
            
            for k in range(pn):
                mask = (idx_w_arr == k)
                if np.any(mask):
                    grad_W[k, :] = 2.0 * np.sum(errs[mask][:, None] * U_sub[mask] * V_sub[mask], axis=0)
            
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            U = U + vel_U
            V = V + vel_V
            W = W + vel_W
            
            if it % 300 == 299:
                lr *= 0.85
                momentum = min(momentum * 0.98, 0.95)
        
        return best_loss, best_U, best_V, best_W
    
    # Try multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: Random Gaussian
    scale = 1.0 / np.sqrt(rank)
    U1 = np.random.randn(nm, rank) * scale
    V1 = np.random.randn(mp, rank) * scale
    W1 = np.random.randn(pn, rank) * scale
    init_strategies.append((U1, V1, W1, 0.02))
    
    # Strategy 2: Near-schoolbook (start close to known solution)
    if rank >= 9:
        U2 = np.zeros((nm, rank))
        V2 = np.zeros((mp, rank))
        W2 = np.zeros((pn, rank))
        # Initialize first min(rank, 18) columns with schoolbook-like structure
        for col in range(min(rank, 18)):
            i = col % 2
            j = (col // 2) % 3
            k = (col // 6) % 3
            U2[i * m + j, col] = 1.0
            V2[j * p + k, col] = 1.0
            W2[k * n + i, col] = 1.0
        # Add small random perturbation to remaining columns
        for col in range(18, rank):
            U2[:, col] = np.random.randn(nm) * 0.1
            V2[:, col] = np.random.randn(mp) * 0.1
            W2[:, col] = np.random.randn(pn) * 0.1
        init_strategies.append((U2, V2, W2, 0.01))
    
    # Strategy 3: Sparse random
    U3 = np.zeros((nm, rank))
    V3 = np.zeros((mp, rank))
    W3 = np.zeros((pn, rank))
    for col in range(rank):
        # Each rank-1 term has sparse support
        u_idx = np.random.choice(nm, size=min(3, nm), replace=False)
        v_idx = np.random.choice(mp, size=min(3, mp), replace=False)
        w_idx = np.random.choice(pn, size=min(3, pn), replace=False)
        U3[u_idx, col] = np.random.choice([-1, 1], size=len(u_idx))
        V3[v_idx, col] = np.random.choice([-1, 1], size=len(v_idx))
        W3[w_idx, col] = np.random.choice([-1, 1], size=len(w_idx))
    init_strategies.append((U3, V3, W3, 0.05))
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for U_init, V_init, W_init, lr in init_strategies:
        loss, U_opt, V_opt, W_opt = _optimize(U_init.copy(), V_init.copy(), W_init.copy(), lr_init=lr)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_opt, V_opt, W_opt
    
    if best_loss < 1e-4:
        # Try rounding to simple fractions
        for mult in [1, 2, 4, 8, 16]:
            U_rounded = np.round(best_U * mult) / mult
            V_rounded = np.round(best_V * mult) / mult
            W_rounded = np.round(best_W * mult) / mult
            
            if _verify_233(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-8):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        if _verify_233(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-8):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _expand_233_to_239(U_233, V_233, W_233):
    """
    Expand a 2x3x3 decomposition to a 2x3x9 decomposition.
    """
    n, m, p = DIMENSIONS
    n_233, m_233, p_233 = DIMENSIONS_233
    r_233 = len(U_233[0])
    
    U_239 = [[0] * (3 * r_233) for _ in range(n * m)]
    V_239 = [[0] * (3 * r_233) for _ in range(m * p)]
    W_239 = [[0] * (3 * r_233) for _ in range(p * n)]
    
    for s in range(3):
        for beta in range(r_233):
            alpha = s * r_233 + beta
            
            # U
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    U_239[idx_u][alpha] = U_233[idx_u][beta]
            
            # V
            for j in range(m):
                for k_prime in range(p_233):
                    k = s * p_233 + k_prime
                    idx_v = j * p + k
                    idx_v_233 = j * p_233 + k_prime
                    V_239[idx_v][alpha] = V_233[idx_v_233][beta]
            
            # W
            for k_prime in range(p_233):
                for i in range(n):
                    k = s * p_233 + k_prime
                    idx_w = k * n + i
                    idx_w_233 = k_prime * n + i
                    W_239[idx_w][alpha] = W_233[idx_w_233][beta]
    
    return U_239, V_239, W_239

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    best_result = _schoolbook()
    best_rank = 54
    
    # Try ranks for 2x3x3 from 9 to 18, with more seeds for lower ranks
    seeds_by_rank = {
        9: [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        10: [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384],
        11: [42, 123, 456, 789, 1024, 2048, 4096, 8192],
        12: [42, 123, 456, 789, 1024, 2048, 4096],
        13: [42, 123, 456, 789, 1024, 2048],
        14: [42, 123, 456, 789, 1024],
        15: [42, 123, 456, 789],
        16: [42, 123, 456],
        17: [42, 123],
        18: [42],
    }
    
    # Allocate more iterations to lower ranks
    iters_by_rank = {
        9: 3000, 10: 2500, 11: 2000, 12: 1500, 13: 1200,
        14: 1000, 15: 800, 16: 600, 17: 500, 18: 400
    }
    
    for r_233 in range(9, 19):
        if time.time() > deadline - 10:
            break
        
        seeds = seeds_by_rank.get(r_233, [42])
        max_iters = iters_by_rank.get(r_233, 1000)
        
        for seed in seeds:
            if time.time() > deadline - 5:
                break
                
            res_233 = _try_rank_233(r_233, max_iters=max_iters, tol=1e-7, seed=seed)
            if res_233 is not None:
                U_233, V_233, W_233 = res_233
                U_239, V_239, W_239 = _expand_233_to_239(U_233, V_233, W_233)
                
                if _verify(U_239, V_239, W_239):
                    rank_239 = 3 * r_233
                    if rank_239 < best_rank:
                        best_result = {"U": U_239, "V": V_239, "W": W_239}
                        best_rank = rank_239
                    break  # Found valid for this rank, move to next (lower) rank
    
    return best_result
