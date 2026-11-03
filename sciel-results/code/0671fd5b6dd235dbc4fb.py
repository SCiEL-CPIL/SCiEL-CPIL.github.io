"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s wall-clock
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    min_rank = max(n * p, m * p, n * m)
    
    # Try structured searches for lower rank
    # Use a combination of structured initialization and local refinement
    
    for target_rank in range(min_rank, min(rank_school, 60)):
        if time.time() > deadline - 15:
            break
        
        # Try multiple structured initializations
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: use block structure
            U, V, W = _structured_init(n, m, p, target_rank, attempt)
            
            if U is None:
                continue
            
            # Verify
            if _verify_fast(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                return best_result
            
            # Try local refinement
            U, V, W = _local_refine(U, V, W, n, m, p, time.time(), deadline - 5)
            
            if _verify_fast(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                return best_result
    
    # Try random search as fallback
    for target_rank in range(min_rank, min(rank_school, 50)):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 3:
                break
            
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n * m, target_rank))
            V = np.random.choice(vals, size=(m * p, target_rank))
            W = np.random.choice(vals, size=(p * n, target_rank))
            
            if _verify_fast(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                return best_result
    
    return best_result

def _structured_init(n, m, p, rank, seed):
    """Create a structured initialization based on block decomposition."""
    np.random.seed(seed)
    
    # Try to create a decomposition that respects the matrix structure
    # Use a mixture of identity-like and random patterns
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: for each "column" c, assign a pattern
    for c in range(rank):
        # Choose which (i,j), (j,k), (k,i) to activate
        # Use a structured pattern based on c
        
        # Pattern 1: diagonal-like
        if c % 3 == 0:
            for i in range(n):
                j = (i + c) % m
                k = (i + c) % p
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
        # Pattern 2: block diagonal
        elif c % 3 == 1:
            i = (c // 3) % n
            j = (c // 3) % m
            k = (c // 3) % p
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        # Pattern 3: random sparse
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            U[i * m + j, c] = random.choice([-1, 1])
            V[j * p + k, c] = random.choice([-1, 1])
            W[k * n + i, c] = random.choice([-1, 1])
    
    return U, V, W

def _local_refine(U, V, W, n, m, p, start_time, deadline):
    """Try to refine the decomposition using local updates."""
    # This is a simplified local search
    # For each column, try flipping signs or changing values
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    for c in range(r):
        if time.time() > deadline:
            break
        
        # Try different values for this column
        best_improvement = None
        best_vals = None
        
        for u_val in [-1, -0.5, 0, 0.5, 1]:
            for v_val in [-1, -0.5, 0, 0.5, 1]:
                for w_val in [-1, -0.5, 0, 0.5, 1]:
                    # Save original
                    orig_u = U[:, c].copy()
                    orig_v = V[:, c].copy()
                    orig_w = W[:, c].copy()
                    
                    # Apply change
                    U[:, c] = u_val * orig_u
                    V[:, c] = v_val * orig_v
                    W[:, c] = w_val * orig_w
                    
                    # Check if this is closer to solution
                    error = _compute_error(U, V, W, n, m, p)
                    
                    if best_improvement is None or error < best_improvement:
                        best_improvement = error
                        best_vals = (u_val, v_val, w_val)
                    
                    # Restore
                    U[:, c] = orig_u
                    V[:, c] = orig_v
                    W[:, c] = orig_w
        
        # Apply best change
        if best_vals is not None:
            u_val, v_val, w_val = best_vals
            U[:, c] = u_val * U[:, c]
            V[:, c] = v_val * V[:, c]
            W[:, c] = w_val * W[:, c]
    
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.sum((T - 1.0) ** 2)

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _try_strassen_based(n, m, p, time_limit):
    """
    Try to find a decomposition using Strassen-like techniques.
    Placeholder.
    """
    pass

# EVOLVE-BLOCK-END
