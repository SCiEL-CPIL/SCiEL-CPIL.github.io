"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (val - 1.0) ** 2
    return error

def _is_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _local_search(U, V, W, n, m, p, max_iter=10000, timeout=5):
    """Local search to minimize error."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0])
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    vals = [-1, -0.5, 0, 0.5, 1]
    current_error = _compute_error(U, V, W, n, m, p)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = current_error
    
    for it in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error < 1e-12:
            break
        
        # Randomly pick a matrix and an entry to change
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            row, col = random.randint(0, nm-1), random.randint(0, rank-1)
            old_val = U[row, col]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            U[row, col] = new_val
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row, col = random.randint(0, mp-1), random.randint(0, rank-1)
            old_val = V[row, col]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            V[row, col] = new_val
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[row, col] = old_val
        else:
            row, col = random.randint(0, pn-1), random.randint(0, rank-1)
            old_val = W[row, col]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            W[row, col] = new_val
            new_error = _compute_error(U, V, W, n, m, p)
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[row, col] = old_val
    
    return best_U, best_V, best_W, best_error

def _try_rank_decomposition(n, m, p, target_rank, timeout=30):
    """Try to find a decomposition of the given rank."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for attempt in range(20):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization
        U = np.random.choice(vals, size=(nm, target_rank))
        V = np.random.choice(vals, size=(mp, target_rank))
        W = np.random.choice(vals, size=(pn, target_rank))
        
        # Local search
        U, V, W, error = _local_search(U, V, W, n, m, p, max_iter=5000, timeout=min(5, timeout - (time.time() - start_time)))
        
        if error < 1e-12:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find a low-rank decomposition
    for rank in range(25, 76):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _try_rank_decomposition(n, m, p, rank, timeout=min(10, remaining))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
