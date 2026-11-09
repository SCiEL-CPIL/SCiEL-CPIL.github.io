"""Matrix multiplication tensor decomposition: 3x4 x 4x8 -> 3x8
Uses a structured search to find a rank-minimal exact tensor decomposition.
Falls back to schoolbook (rank 96) if no improvement is found within time budget.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum absolute error of the tensor decomposition."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(r, n, m, p, deadline, seed_offset=0):
    """Try to find a valid decomposition with rank r using local search."""
    random.seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)
    
    # Initialize with random values from {-1, 0, 1}
    U = np.random.randint(-1, 2, size=(n*m, r))
    V = np.random.randint(-1, 2, size=(m*p, r))
    W = np.random.randint(-1, 2, size=(p*n, r))
    
    best_err = _compute_error(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        if step % 1000 == 0 and _compute_error(best_U, best_V, best_W, n, m, p) < 1e-10:
            return best_U, best_V, best_W
        
        # Randomly pick one matrix and one entry to modify
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            row = random.randint(0, n*m-1)
            col = random.randint(0, r-1)
            old_val = U[row, col]
            new_val = random.randint(-2, 2)
            if new_val == old_val:
                continue
            U[row, col] = new_val
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row = random.randint(0, m*p-1)
            col = random.randint(0, r-1)
            old_val = V[row, col]
            new_val = random.randint(-2, 2)
            if new_val == old_val:
                continue
            V[row, col] = new_val
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[row, col] = old_val
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, r-1)
            old_val = W[row, col]
            new_val = random.randint(-2, 2)
            if new_val == old_val:
                continue
            W[row, col] = new_val
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[row, col] = old_val
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
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

def construct():
    start_time = time.time()
    DEADLINE = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Try to find a decomposition with rank < 96
    for r in [95, 94, 93, 92, 90, 88, 84, 80, 72, 64, 48]:
        if time.time() > DEADLINE - 20:
            break
        result = _try_rank(r, n, m, p, DEADLINE - 10, seed_offset=r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
