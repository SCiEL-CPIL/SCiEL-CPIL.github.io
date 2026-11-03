import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target(n, m, p):
    """Build target: T[i,j,k] = 1 for all valid (i,j,k)"""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_exact_fast(U, V, W, n, m, p):
    """Fast exact verification using direct sum computation.
    Returns True if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _verify_exact_int(U, V, W, n, m, p):
    """Exact verification assuming integer/half-integer values.
    Multiply by 2 to work with integers."""
    rank = U.shape[1]
    U2 = np.round(U * 2)
    V2 = np.round(V * 2)
    W2 = np.round(W * 2)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += int(U2[i * m + j, r]) * int(V2[j * p + k, r]) * int(W2[k * n + i, r])
                # total should equal 8 (since 1 = 8/8, and we multiplied each by 2, so product is 8x)
                if total != 8:
                    return False
    return True

def _local_search_fast(U, V, W, n, m, p, values, max_iter=5000, timeout=5.0):
    """Local search using fast verification. Returns (success, U, V, W)."""
    start_time = time.time()
    rank = U.shape[1]
    
    # Check if already exact
    if _verify_exact_fast(U, V, W, n, m, p):
        return True, U.copy(), V.copy(), W.copy()
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        # Pick random matrix and position
        mat_choice = random.choice([0, 1, 2])
        if mat_choice == 0:
            mat = U
            idx = random.randint(0, U.size - 1)
        elif mat_choice == 1:
            mat = V
            idx = random.randint(0, V.size - 1)
        else:
            mat = W
            idx = random.randint(0, W.size - 1)
        
        old_val = mat.flat[idx]
        new_val = random.choice(values)
        
        mat.flat[idx] = new_val
        if _verify_exact_fast(U, V, W, n, m, p):
            return True, U.copy(), V.copy(), W.copy()
        mat.flat[idx] = old_val
    
    return False, U.copy(), V.copy(), W.copy()

def _try_rank_structured(R, n, m, p, values, time_limit):
    """Try to find rank-R decomposition using structured initialization + local search."""
    start_time = time.time()
    
    # Strategy 1: Try structured patterns based on known algorithms
    # For 2x3x4, we can try patterns inspired by matrix multiplication
    
    # Initialize with zeros
    U0 = np.zeros((n * m, R), dtype=np.float64)
    V0 = np.zeros((m * p, R), dtype=np.float64)
    W0 = np.zeros((p * n, R), dtype=np.float64)
    
    # Try several structured initializations
    best_result = None
    
    # Pattern 1: Random sparse initialization
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        
        # Place random non-zero entries
        num_nonzero_u = random.randint(R, min(n * m * R, 2 * R))
        num_nonzero_v = random.randint(R, min(m * p * R, 2 * R))
        num_nonzero_w = random.randint(R, min(p * n * R, 2 * R))
        
        for _ in range(num_nonzero_u):
            r = random.randint(0, R - 1)
            idx = random.randint(0, n * m - 1)
            U[idx, r] = random.choice([1.0, -1.0, 0.5, -0.5])
        for _ in range(num_nonzero_v):
            r = random.randint(0, R - 1)
            idx = random.randint(0, m * p - 1)
            V[idx, r] = random.choice([1.0, -1.0, 0.5, -0.5])
        for _ in range(num_nonzero_w):
            r = random.randint(0, R - 1)
            idx = random.randint(0, p * n - 1)
            W[idx, r] = random.choice([1.0, -1.0, 0.5, -0.5])
        
        # Local search
        remaining = time_limit - (time.time() - start_time)
        if remaining < 0.5:
            break
        success, U, V, W = _local_search_fast(U, V, W, n, m, p, values, 
                                               max_iter=2000, timeout=min(remaining, 3.0))
        if success:
            return U, V, W
    
    # Pattern 2: Try to use the fact that we need sum to be 1
    # Initialize with values that might sum to 1
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        
        # For each rank index, assign values
        for r in range(R):
            # Randomly choose which positions get non-zero values
            u_positions = set(random.sample(range(n * m), random.randint(1, 3)))
            v_positions = set(random.sample(range(m * p), random.randint(1, 3)))
            w_positions = set(random.sample(range(p * n), random.randint(1, 3)))
            
            for pos in u_positions:
                U[pos, r] = random.choice([1.0, -1.0])
            for pos in v_positions:
                V[pos, r] = random.choice([1.0, -1.0])
            for pos in w_positions:
                W[pos, r] = random.choice([1.0, -1.0])
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 0.5:
            break
        success, U, V, W = _local_search_fast(U, V, W, n, m, p, values,
                                               max_iter=2000, timeout=min(remaining, 3.0))
        if success:
            return U, V, W
    
    return None

def _try_rank_gradient(R, n, m, p, values, time_limit):
    """Try to find rank-R decomposition using gradient descent + rounding."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    # Target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    U_flat_size = n * m * R
    V_flat_size = m * p * R
    W_flat_size = p * n * R
    
    def objective(x):
        U_r = x[:U_flat_size].reshape(n * m, R)
        V_r = x[U_flat_size:U_flat_size + V_flat_size].reshape(m * p, R)
        W_r = x[U_flat_size + V_flat_size:].reshape(p * n, R)
        recon = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        return np.sum((recon - T) ** 2)
    
    def gradient(x):
        U_r = x[:U_flat_size].reshape(n * m, R)
        V_r = x[U_flat_size:U_flat_size + V_flat_size].reshape(m * p, R)
        W_r = x[U_flat_size + V_flat_size:].reshape(p * n, R)
        recon = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        err = recon - T
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', err, V_r, W_r)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', err, U_r, W_r)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', err, U_r, V_r)
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    # Try a few restarts
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
        
        U_real = np.random.randn(n * m, R) * 0.3
        V_real = np.random.randn(m * p, R) * 0.3
        W_real = np.random.randn(p * n, R) * 0.3
        x0 = np.concatenate([U_real.flatten(), V_real.flatten(), W_real.flatten()])
        
        try:
            opt_res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 50, 'maxfun': 100, 'disp': False})
        except Exception:
            continue
        
        x_opt = opt_res.x
        U_opt = x_opt[:U_flat_size].reshape(n * m, R)
        V_opt = x_opt[U_flat_size:U_flat_size + V_flat_size].reshape(m * p, R)
        W_opt = x_opt[U_flat_size + V_flat_size:].reshape(p * n, R)
        
        # Round to half-integers
        U_int = np.round(U_opt * 2) / 2.0
        V_int = np.round(V_opt * 2) / 2.0
        W_int = np.round(W_opt * 2) / 2.0
        
        # Verify
        if _verify_exact_fast(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
        
        # Try local search
        remaining = time_limit - (time.time() - start_time)
        if remaining > 1.0:
            success, U_int, V_int, W_int = _local_search_fast(
                U_int, V_int, W_int, n, m, p, values,
                max_iter=3000, timeout=min(remaining, 5.0))
            if success:
                return U_int, V_int, W_int
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    # Values for local search
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Known upper bound for 2x3x4 matrix multiplication rank is 18
    # Try ranks from 14 to 23 (schoolbook is 24)
    for R in range(14, best_rank):
        if time.time() - start_time > deadline:
            break
        
        ranks_remaining = best_rank - R
        time_budget = (deadline - (time.time() - start_time)) / max(1, ranks_remaining)
        time_budget = min(time_budget, 15.0)  # Cap per rank
        
        if time_budget < 1.0:
            break
        
        # Try structured search first (faster)
        result = _try_rank_structured(R, n, m, p, values, time_budget * 0.6)
        if result is not None:
            U, V, W = result
            if _verify_exact_fast(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
        
        # Try gradient-based search with remaining time
        remaining = deadline - (time.time() - start_time)
        if remaining > 2.0:
            result = _try_rank_gradient(R, n, m, p, values, remaining * 0.4)
            if result is not None:
                U, V, W = result
                if _verify_exact_fast(U, V, W, n, m, p):
                    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    best_rank = R
                    break
    
    return best_result
