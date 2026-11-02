import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _reconstruct(U, V, W):
    return np.einsum('ur,vr,wr->uvw', U, V, W)

def _error(U, V, W, T):
    return np.linalg.norm(_reconstruct(U, V, W) - T)

def _round_to_half_integer(x):
    return np.round(x * 2.0) / 2.0

def _gradient_step(U, V, W, T, lr, n, m, p):
    """One gradient descent step on continuous relaxation."""
    R = U.shape[1]
    # Gradient of ||U⊗V⊗W - T||^2 w.r.t. U
    # d/dU = 2 * (recon - T) contracted with V and W
    recon = _reconstruct(U, V, W)
    diff = recon - T  # (n*m, m*p, p*n)
    
    # grad_U[u, r] = sum_{v,w} diff[u,v,w] * V[v,r] * W[w,r]
    grad_U = np.einsum('uvw,vr,wr->ur', diff, V, W)
    grad_V = np.einsum('uvw,ur,wr->vr', diff, U, W)
    grad_W = np.einsum('uvw,ur,vr->wr', diff, U, V)
    
    U -= lr * grad_U
    V -= lr * grad_V
    W -= lr * grad_W
    return U, V, W

def _coordinate_descent(U, V, W, T, values, max_iter, timeout):
    """Coordinate descent with exact error, trying discrete values."""
    start_time = time.time()
    current_err = _error(U, V, W, T)
    
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    for it in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_err < 1e-12:
            break
        
        best_improvement = 0
        best_mat = None
        best_idx = None
        best_new_val = None
        best_old_val = None
        best_new_err = current_err
        
        # Try a batch of random variables
        batch = min(50, total_vars)
        for _ in range(batch):
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < n_u:
                mat, idx = U, var_idx
            elif var_idx < n_u + n_v:
                mat, idx = V, var_idx - n_u
            else:
                mat, idx = W, var_idx - n_u - n_v
            
            old_val = mat.flat[idx]
            for new_val in values:
                if abs(new_val - old_val) < 1e-9:
                    continue
                mat.flat[idx] = new_val
                e = _error(U, V, W, T)
                if e < best_new_err:
                    best_new_err = e
                    best_improvement = current_err - e
                    best_mat, best_idx = mat, idx
                    best_new_val, best_old_val = new_val, old_val
                else:
                    mat.flat[idx] = old_val
                    break
            else:
                # No value improved, revert to original
                mat.flat[idx] = old_val
                continue
            
            # Revert to original for next candidate
            mat.flat[idx] = old_val
        
        if best_mat is not None:
            best_mat.flat[best_idx] = best_new_val
            current_err = best_new_err
        else:
            break
    
    return current_err

def _try_rank(R, n, m, p, T, time_limit):
    """Try to find rank-R decomposition."""
    start_time = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    num_restarts = 8
    time_per_restart = time_limit / num_restarts
    
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random structure: start from a partial schoolbook
        # and perturb
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Fill with random half-integers
        for col in range(R):
            # Random pattern
            for i in range(n):
                for j in range(m):
                    U[i*m+j, col] = random.choice([-1.0, 0.0, 1.0])
            for j in range(m):
                for k in range(p):
                    V[j*p+k, col] = random.choice([-1.0, 0.0, 1.0])
            for k in range(p):
                for i in range(n):
                    W[k*n+i, col] = random.choice([-1.0, 0.0, 1.0])
        
        # Continuous gradient refinement
        remaining = time_per_restart - (time.time() - start_time)
        if remaining > 0.1:
            grad_budget = remaining * 0.5
            grad_start = time.time()
            lr = 0.01
            for step in range(5000):
                if time.time() - grad_start > grad_budget:
                    break
                U, V, W = _gradient_step(U, V, W, T, lr, n, m, p)
                # Adaptive learning rate
                if step % 500 == 0 and step > 0:
                    lr *= 0.5
        
        # Round to nearest half-integer
        U = _round_to_half_integer(U)
        V = _round_to_half_integer(V)
        W = _round_to_half_integer(W)
        
        err = _error(U, V, W, T)
        
        # Coordinate descent polish
        remaining = time_per_restart - (time.time() - start_time)
        if remaining > 0.1:
            err = _coordinate_descent(U, V, W, T, values, max_iter=5000, timeout=remaining)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-12:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook fallback: rank 24
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Search ranks from 18 to 23
    for R in range(18, rank_schoolbook):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining = deadline - elapsed
        # Allocate time proportional to remaining ranks
        ranks_left = rank_schoolbook - R
        time_budget = remaining / ranks_left
        time_budget = min(time_budget, 25.0)
        
        if time_budget < 1.5:
            break
        
        result = _try_rank(R, n, m, p, T, time_budget)
        
        if result is not None:
            U, V, W = result
            err = _error(U, V, W, T)
            if err < 1e-12:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
