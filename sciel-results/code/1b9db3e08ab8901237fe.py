import time
import numpy as np

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # For each valid (i,j,k), check that the sum equals 1
    # And for invalid combinations, check that sum equals 0
    
    # Build index arrays
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # val = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                # = sum_r (U[u_idx, r] * V[v_idx, r]) * W[w_idx, r]
                uv = U[u_idx] * V[v_idx]
                val = np.dot(uv, W[w_idx])
                
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal: for each (u_idx, v_idx, w_idx) where indices don't match
    # i = u_idx//m, j = u_idx%m
    # j2 = v_idx//p, k = v_idx%p  
    # k2 = w_idx//n, i2 = w_idx%n
    # Need i==i2, j==j2, k==k2 for valid
    
    for u_idx in range(n * m):
        i = u_idx // m
        j = u_idx % m
        for v_idx in range(m * p):
            j2 = v_idx // p
            k = v_idx % p
            for w_idx in range(p * n):
                k2 = w_idx // n
                i2 = w_idx % n
                
                if i == i2 and j == j2 and k == k2:
                    continue  # Already checked
                
                uv = U[u_idx] * V[v_idx]
                val = np.dot(uv, W[w_idx])
                if abs(val) > tol:
                    return False
    
    return True

def _loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for the decomposition."""
    rank = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                uv = U[u_idx] * V[v_idx]
                val = np.dot(uv, W[w_idx])
                diff = val - 1.0
                loss += diff * diff
                
                # Gradients
                grad_U[u_idx] += 2 * diff * (V[v_idx] * W[w_idx])
                grad_V[v_idx] += 2 * diff * (U[u_idx] * W[w_idx])
                grad_W[w_idx] += 2 * diff * (U[u_idx] * V[v_idx])
    
    # Off-diagonal terms
    for u_idx in range(n * m):
        i = u_idx // m
        j = u_idx % m
        for v_idx in range(m * p):
            j2 = v_idx // p
            k = v_idx % p
            for w_idx in range(p * n):
                k2 = w_idx // n
                i2 = w_idx % n
                
                if i == i2 and j == j2 and k == k2:
                    continue
                
                uv = U[u_idx] * V[v_idx]
                val = np.dot(uv, W[w_idx])
                loss += val * val
                
                grad_U[u_idx] += 2 * val * (V[v_idx] * W[w_idx])
                grad_V[v_idx] += 2 * val * (U[u_idx] * W[w_idx])
                grad_W[w_idx] += 2 * val * (U[u_idx] * V[v_idx])
    
    return loss, grad_U, grad_V, grad_W

def _optimize_decomposition(n, m, p, rank, seed=42, max_iters=3000):
    """Optimize for a rank-r decomposition."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    lr = 0.005
    momentum = 0.9
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        loss, gU, gV, gW = _loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_loss < 1e-12:
                break
        
        vel_U = momentum * vel_U - lr * gU
        vel_V = momentum * vel_V - lr * gV
        vel_W = momentum * vel_W - lr * gW
        
        U += vel_U
        V += vel_V
        W += vel_W
        
        if it % 50 == 0 and it > 0:
            if loss < best_loss * 0.95:
                lr *= 1.05
            else:
                lr *= 0.95
            lr = max(lr, 1e-7)
    
    if best_loss < 1e-10:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _construct_block_strassen(n, m, p):
    """Original block Strassen construction - rank 42."""
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            col += 1
    
    return U, V, W

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    # Start from rank 30 and go up to 41
    for rank in range(30, 42):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        rank_budget = min(15, TIME_LIMIT - 20 - (time.time() - START_TIME))
        if rank_budget < 3:
            break
        
        # Try a few seeds
        for seed in range(3):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            iters = max(500, int(2000 * rank_budget / 15))
            result = _optimize_decomposition(n, m, p, rank, seed=seed, max_iters=iters)
            
            if result is not None:
                U, V, W = result
                if _verify_decomposition_fast(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: use the known rank-42 solution
    U, V, W = _construct_block_strassen(n, m, p)
    
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
    
    return {"U": U, "V": V, "W": W}
