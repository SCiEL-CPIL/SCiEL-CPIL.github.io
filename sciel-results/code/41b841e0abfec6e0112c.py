import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target: sum_r U[i,j,r] V[j,k,r] W[k',i',r] = delta_{i,i'} delta_{k,k'}
    # We check for all i, j, k, i', k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        total = 0.0
                        for c in range(rank):
                            total += U[i*m+j][c] * V[j*p+k][c] * W[kp*n+ip][c]
                        target = 1.0 if (i == ip and k == kp) else 0.0
                        if abs(total - target) > 1e-9:
                            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term A[i,j] B[j,k] contributes to C[i,k]
                # So U[i*m+j, col] = 1
                # V[j*p+k, col] = 1
                # W[k*n+i, col] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_error(U, V, W, n, m, p):
    """Compute the Frobenius norm of the error tensor."""
    rank = len(U[0]) if U else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Error E[i,j,k,ip,kp] = Target - Sum
    # We can compute the sum S[ip,kp] for each i,j,k
    # S[ip,kp] = sum_c U[i,j,c] V[j,k,c] W[kp,ip,c]
    # Target is 1 if i==ip and k==kp else 0.
    
    # To be fast, we can vectorize or just loop.
    # n=2, m=5, p=5.
    # Total terms 2*5*5 = 50.
    # For each term, we compute a 2x5 matrix S.
    # Sum of squares of (S - Target).
    
    err_sq = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute S[ip][kp]
                S = np.zeros((n, p))
                for c in range(rank):
                    u_val = U[i*m+j][c]
                    v_val = V[j*p+k][c]
                    if u_val == 0 or v_val == 0:
                        continue
                    # W is flattened k*n+ip
                    # We need W[kp*n+ip][c]
                    # Loop over kp, ip
                    for kp in range(p):
                        for ip in range(n):
                            S[ip, kp] += u_val * v_val * W[kp*n+ip][c]
                
                # Compute error for this i,j,k
                for ip in range(n):
                    for kp in range(p):
                        target = 1.0 if (i == ip and k == kp) else 0.0
                        diff = S[ip, kp] - target
                        err_sq += diff * diff
    return np.sqrt(err_sq)

def _try_rank_local_search(n, m, p, rank, time_limit=10.0, seed=42):
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    values = [0, 0.5, -0.5, 1, -1]
    
    start_time = time.time()
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = [[rng.choice(values) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.choice(values) for _ in range(rank)] for _ in range(mp)]
        W = [[rng.choice(values) for _ in range(rank)] for _ in range(pn)]
        
        current_err = _compute_error(U, V, W, n, m, p)
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            if current_err < best_err:
                best_err = current_err
                best_solution = (U, V, W)
                
            if current_err < 1e-9:
                return {"U": U, "V": V, "W": W}
            
            # Pick a random variable and try changing it
            var_type = rng.randrange(3) # 0:U, 1:V, 2:W
            if var_type == 0:
                idx = rng.randrange(nm)
                c = rng.randrange(rank)
                old_val = U[idx][c]
                new_val = rng.choice(values)
                if new_val == old_val: continue
                U[idx][c] = new_val
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err < current_err:
                    current_err = new_err
                    improved = True
                else:
                    U[idx][c] = old_val
            elif var_type == 1:
                idx = rng.randrange(mp)
                c = rng.randrange(rank)
                old_val = V[idx][c]
                new_val = rng.choice(values)
                if new_val == old_val: continue
                V[idx][c] = new_val
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err < current_err:
                    current_err = new_err
                    improved = True
                else:
                    V[idx][c] = old_val
            else:
                idx = rng.randrange(pn)
                c = rng.randrange(rank)
                old_val = W[idx][c]
                new_val = rng.choice(values)
                if new_val == old_val: continue
                W[idx][c] = new_val
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err < current_err:
                    current_err = new_err
                    improved = True
                else:
                    W[idx][c] = old_val
    
    if best_solution and best_err < 1e-9:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank
    # Start from 49 down to 25
    for rank in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_local_search(n, m, p, rank, time_limit=5.0, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
