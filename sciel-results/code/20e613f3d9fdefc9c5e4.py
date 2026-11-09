# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find better rank
    for r in range(max(n*m, m*p, p*n), schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        # Try local search
        result = _local_search(n, m, p, r, deadline - 5)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best

def _local_search(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using local search."""
    rng = np.random.RandomState(42)
    
    # Initialize with schoolbook-like structure
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Try different initializations
    for init in range(3):
        if time.time() > deadline:
            break
        
        if init == 0:
            # Sparse random
            for s in range(r):
                U[rng.randint(n*m), s] = rng.choice([-1, 1])
                V[rng.randint(m*p), s] = rng.choice([-1, 1])
                W[rng.randint(p*n), s] = rng.choice([-1, 1])
        elif init == 1:
            # Structured
            for s in range(r):
                i = s % n
                j = (s // n) % m
                k = (s // (n*m)) % p
                if i*m+j < n*m:
                    U[i*m+j, s] = 1
                if j*p+k < m*p:
                    V[j*p+k, s] = 1
                if k*n+i < p*n:
                    W[k*n+i, s] = 1
        else:
            # Random from small set
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
        
        # Local search
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            # Compute error
            error = _compute_error(U, V, W, n, m, p)
            if error < 1e-9:
                # Verify and return
                if _verify_numpy(U, V, W, n, m, p):
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            
            # Try to improve
            improved = False
            for _ in range(10):
                # Randomly pick a matrix and entry
                mat_idx = rng.randint(3)
                if mat_idx == 0:
                    row, col = rng.randint(n*m), rng.randint(r)
                    old_val = U[row, col]
                    new_val = rng.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val != old_val:
                        U[row, col] = new_val
                        new_error = _compute_error(U, V, W, n, m, p)
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            U[row, col] = old_val
                elif mat_idx == 1:
                    row, col = rng.randint(m*p), rng.randint(r)
                    old_val = V[row, col]
                    new_val = rng.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val != old_val:
                        V[row, col] = new_val
                        new_error = _compute_error(U, V, W, n, m, p)
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            V[row, col] = old_val
                else:
                    row, col = rng.randint(p*n), rng.randint(r)
                    old_val = W[row, col]
                    new_val = rng.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val != old_val:
                        W[row, col] = new_val
                        new_error = _compute_error(U, V, W, n, m, p)
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            W[row, col] = old_val
                
                if improved:
                    break
            
            if not improved:
                break
    
    return None

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                error += (val - expected) ** 2
    return error

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-END
