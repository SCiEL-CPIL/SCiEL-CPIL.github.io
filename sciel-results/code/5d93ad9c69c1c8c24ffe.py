import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _get_schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_error(U, V, W, T):
    """Compute the error between the tensor product sum and T."""
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    error = np.linalg.norm(reconstructed - T)
    return error

def _check_exact(U, V, W, n, m, p):
    """
    Check if the decomposition is exact for all (i,j,k) positions and zero elsewhere.
    """
    # Check all target positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all non-target positions
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_set.add((u, v, w))
    
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in target_set:
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if abs(val) > 1e-9:
                        return False
    return True

def _try_rank_with_search(R, n, m, p, T, time_limit):
    """
    Try to find a decomposition of rank R using structured search.
    """
    start_time = time.time()
    values = [-1, 0, 1, -0.5, 0.5]
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 30
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random values, but ensure some structure
        U = np.random.choice(values, size=(n * m, R))
        V = np.random.choice(values, size=(m * p, R))
        W = np.random.choice(values, size=(p * n, R))
        
        # Ensure at least one non-zero in each column for better starting point
        for r in range(R):
            if np.all(U[:, r] == 0):
                U[random.randint(0, n*m-1), r] = random.choice(values)
            if np.all(V[:, r] == 0):
                V[random.randint(0, m*p-1), r] = random.choice(values)
            if np.all(W[:, r] == 0):
                W[random.randint(0, p*n-1), r] = random.choice(values)
        
        current_error = _compute_error(U, V, W, T)
        
        # Coordinate descent local search
        max_iter = 3000
        no_improve = 0
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if current_error < 1e-12:
                break
            
            # Pick a random variable to change
            total_vars = U.size + V.size + W.size
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < U.size:
                mat = U
                idx = var_idx
            elif var_idx < U.size + V.size:
                mat = V
                idx = var_idx - U.size
            else:
                mat = W
                idx = var_idx - U.size - V.size
                
            old_val = mat.flat[idx]
            
            # Try multiple candidate values and pick the best
            best_new_val = old_val
            best_new_error = current_error
            
            for candidate in values:
                if candidate == old_val:
                    continue
                mat.flat[idx] = candidate
                new_error = _compute_error(U, V, W, T)
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_new_val = candidate
            
            if best_new_val != old_val:
                mat.flat[idx] = best_new_val
                current_error = best_new_error
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    # Perturb randomly to escape local minimum
                    for _ in range(5):
                        pi = random.randint(0, total_vars - 1)
                        if pi < U.size:
                            mat = U
                            idx = pi
                        elif pi < U.size + V.size:
                            mat = V
                            idx = pi - U.size
                        else:
                            mat = W
                            idx = pi - U.size - V.size
                        mat.flat[idx] = random.choice(values)
                    current_error = _compute_error(U, V, W, T)
                    no_improve = 0
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < 1e-12:
            break
    
    if best_error < 1e-10:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    for R in range(12, best_rank):
        if time.time() - start_time > deadline:
            break
            
        time_budget = min(15.0, (deadline - (time.time() - start_time)) / max(best_rank - R, 1))
        if time_budget < 1.0:
            break
            
        result = _try_rank_with_search(R, n, m, p, T, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify exactness
            if _check_exact(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
