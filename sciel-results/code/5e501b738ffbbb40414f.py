"""Rewritten program for 3x5x7 matrix multiplication tensor rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N
TOTAL = N*M*P

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # (rank,)
            for k in range(p):
                val = np.sum(u_row * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _is_halfint(arr, tol=1e-8):
    """Check if all entries are within tol of half-integers."""
    doubled = arr * 2.0
    nearest = np.round(doubled)
    return np.max(np.abs(doubled - nearest)) < tol

def _loss(U, V, W, n, m, p):
    """Compute loss only (faster than loss+grads)."""
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

def _optimize_als(U, V, W, n, m, p, max_iters=200, tol=1e-10, time_limit=None):
    """Alternating Least Squares optimization.
    
    For each (i,j,k), we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    Fix V and W, solve for U:
    For each i*m+j: sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for each k
    This gives p equations in rank unknowns.
    
    Similarly for V and W.
    """
    rank = U.shape[1]
    best_loss = _loss(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                # For each k, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Build system: A @ x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=1
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k] = V[j*p+k] * W[k*n+i]
                b = np.ones(p)
                # Least squares solve
                try:
                    U[i*m+j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                # For each i, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
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
                # For each j, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
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

def _optimize_grad(U, V, W, n, m, p, max_iters=1000, lr_init=0.01, tol=1e-10, time_limit=None):
    """Gradient descent optimization with adaptive learning rate."""
    rank = U.shape[1]
    best_loss = _loss(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    prev_loss = float('inf')
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Compute gradients
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
        
        # Adaptive learning rate
        if loss > prev_loss:
            lr *= 0.5
        elif it > 0 and loss < prev_loss * 0.9:
            lr = min(lr * 1.05, 0.1)
        prev_loss = loss
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_rank(r, n, m, p, max_seeds=5, time_limit=None):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    if time_limit is not None and time.time() > time_limit:
        return None
    
    rng = np.random.RandomState(42)
    scale = 1.0 / np.sqrt(r)
    
    for seed in range(max_seeds):
        if time_limit is not None and time.time() > time_limit:
            return None
        
        U = rng.randn(n*m, r) * scale
        V = rng.randn(m*p, r) * scale
        W = rng.randn(p*n, r) * scale
        
        # First use ALS
        U, V, W, loss = _optimize_als(U, V, W, n, m, p, max_iters=100, time_limit=time_limit)
        
        if loss < 1e-6:
            U_h = _round_to_halfint(U)
            V_h = _round_to_halfint(V)
            W_h = _round_to_halfint(W)
            if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                return U_h, V_h, W_h
        
        # Polish with gradient descent
        if loss < 1e-3:
            U, V, W, loss2 = _optimize_grad(U, V, W, n, m, p, max_iters=500, lr_init=0.005, time_limit=time_limit)
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
    deadline = start_time + 140  # Safety margin before 150s limit
    
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
    # Start with ranks that are more likely to be achievable
    candidate_ranks = list(range(rank_school - 1, rank_school - 30, -1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        if r >= best_rank:
            continue
        
        result = _try_rank(r, n, m, p, max_seeds=3, time_limit=deadline - 5)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # If we found a good rank, try to go lower
            if best_rank <= 100:
                # Adjust candidate ranks to continue lower
                candidate_ranks = list(range(best_rank - 1, max(34, best_rank - 20), -1))
    
    return best_result
