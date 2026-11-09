"""
Rewritten program for 3x5x7 matrix multiplication tensor rank minimization.
Combines the structural ALS/GD optimization from the current program with 
improved initialization strategies and search heuristics from inspirations.
"""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Vectorized check: 
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a bit tricky to fully vectorize without large intermediate tensors, 
    # but for small n,m,p, the loop is fine.
    
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]
            for k in range(p):
                val = np.sum(u_row * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _loss(U, V, W, n, m, p):
    """Compute loss."""
    rank = U.shape[1]
    loss = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]
            for k in range(p):
                val = np.sum(u_row * V[j*p+k] * W[k*n+i])
                err = val - 1.0
                loss += err * err
    return loss

def _optimize_als(U, V, W, n, m, p, max_iters=50, tol=1e-10, time_limit=None):
    """Alternating Least Squares optimization."""
    rank = U.shape[1]
    best_loss = _loss(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k] = V[j*p+k] * W[k*n+i]
                b = np.ones(p)
                try:
                    U[i*m+j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i] = U[i*m+j] * W[k*n+i]
                b = np.ones(n)
                try:
                    V[j*p+k] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j] = U[i*m+j] * V[j*p+k]
                b = np.ones(m)
                try:
                    W[k*n+i] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        loss = _loss(U, V, W, n, m, p)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def _optimize_grad(U, V, W, n, m, p, max_iters=200, lr_init=0.01, tol=1e-10, time_limit=None):
    """Gradient descent optimization with adaptive learning rate."""
    rank = U.shape[1]
    best_loss = _loss(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    prev_loss = float('inf')
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    val = np.sum(u_row * v_row * w_row)
                    err = val - 1.0
                    loss += err * err
                    
                    grad_U[i*m+j] += err * v_row * w_row
                    grad_V[j*p+k] += err * u_row * w_row
                    grad_W[k*n+i] += err * u_row * v_row
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        if loss > prev_loss:
            lr *= 0.5
        elif it > 0 and loss < prev_loss * 0.9:
            lr = min(lr * 1.05, 0.1)
        prev_loss = loss
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_rank(r, n, m, p, max_seeds=10, time_limit=None):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    if time_limit is not None and time.time() > time_limit:
        return None
    
    # Use different types of initialization to escape local minima
    # 1. Random Gaussian
    # 2. Sparse Random (many zeros)
    # 3. Small Integer/Half-Integer random
    
    for seed in range(max_seeds):
        if time_limit is not None and time.time() > time_limit:
            return None
        
        rng = np.random.RandomState(seed * 1000 + r)
        
        # Strategy: Mix of dense and sparse
        if seed % 3 == 0:
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
        elif seed % 3 == 1:
            # Sparse
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
            mask_u = rng.rand(n*m, r) < 0.5
            mask_v = rng.rand(m*p, r) < 0.5
            mask_w = rng.rand(p*n, r) < 0.5
            U *= mask_u
            V *= mask_v
            W *= mask_w
        else:
            # Small integers/half-integers
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
        
        # Optimize
        U, V, W, loss = _optimize_als(U, V, W, n, m, p, max_iters=20, time_limit=time_limit)
        
        if loss < 1e-4:
            # Polish
            U, V, W, loss2 = _optimize_grad(U, V, W, n, m, p, max_iters=100, lr_init=0.005, time_limit=time_limit)
            if loss2 < 1e-8:
                U_h = _round_to_halfint(U)
                V_h = _round_to_halfint(V)
                W_h = _round_to_halfint(W)
                if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                    return U_h, V_h, W_h
    
    return None

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition (rank = n*m*p = 105)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Try ranks from just below schoolbook down
    # We limit the search depth to ensure we finish in time.
    # If 104-100 fail quickly, we might not try lower.
    candidate_ranks = list(range(rank_school - 1, max(rank_school - 15, 30), -1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, n, m, p, max_seeds=5, time_limit=deadline - 5)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # If we found a good rank, we can try to go lower, but be careful with time
            # Reset candidate ranks to continue lower from the new best
            if best_rank > 30:
                candidate_ranks = list(range(best_rank - 1, max(best_rank - 10, 30), -1))
            else:
                break
    
    return best_result
