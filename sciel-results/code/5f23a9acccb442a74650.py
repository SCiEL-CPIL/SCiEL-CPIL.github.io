import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_list(arr):
    return np.array(arr).tolist()

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank(target_rank, time_budget, n=2, m=3, p=3, seed=42):
    """
    Search for a decomposition of the given rank using randomized linear algebra.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    rng = np.random.default_rng(seed)
    
    num_eqs = n * m * p
    num_unk = p * n * target_rank
    
    # Entry sets for U and V
    entry_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2]
    ]
    
    trial = 0
    while time.time() < deadline:
        trial += 1
        # Determine entry set based on trial count
        es = entry_sets[0] if trial % 2 == 0 else entry_sets[1]
        
        # Generate random U and V
        U = rng.choice(es, size=(n*m, target_rank)).astype(float)
        V = rng.choice(es, size=(m*p, target_rank)).astype(float)
        
        # Ensure columns are not zero
        for c in range(target_rank):
            if np.all(U[:, c] == 0):
                U[0, c] = 1
            if np.all(V[:, c] == 0):
                V[0, c] = 1
        
        # Build linear system A * w = b
        # A is (num_eqs, num_unk)
        # b is (num_eqs,)
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            w_row = k * n + i
                            w_col = c
                            unk_idx = w_row * target_rank + w_col
                            A[eq_idx, unk_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-6:
                continue
        except np.linalg.LinAlgError:
            continue
        
        W_mat = W_vec.reshape(p*n, target_rank)
        
        # Check for integer or half-integer solution
        # Try rounding to nearest integer
        W_int = np.round(W_mat)
        if np.max(np.abs(W_mat - W_int)) < 1e-6:
            if _verify(U, V, W_int):
                return {
                    "U": _to_list(U.astype(int)),
                    "V": _to_list(V.astype(int)),
                    "W": _to_list(W_int.astype(int))
                }
        
        # Try rounding 2*W to nearest integer (half-integers)
        W_half = np.round(2 * W_mat) / 2.0
        if np.max(np.abs(W_mat - W_half)) < 1e-6:
            if _verify(U, V, W_half):
                return {
                    "U": _to_list(U.astype(int)),
                    "V": _to_list(V.astype(int)),
                    "W": _to_list(W_half)
                }
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Lower bound is roughly 12-14 for this size, schoolbook is 18
    # We will try 14, 15, 16, 17
    
    target_ranks = [14, 15, 16, 17]
    
    for r in target_ranks:
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time to this rank
        # Give more time to lower ranks as they are harder
        time_alloc = min(remaining * 0.5, 30.0)
        
        result = _search_rank(r, time_alloc, 2, 3, 3, seed=int(time.time()*1000) % 10000)
        if result is not None:
            return result
        
        # If first attempt fails, try a second attempt with different seed if time permits
        remaining = deadline - time.time()
        if remaining > 10.0:
            time_alloc = min(remaining * 0.5, 30.0)
            result = _search_rank(r, time_alloc, 2, 3, 3, seed=int(time.time()*1000) % 10000 + 1)
            if result is not None:
                return result

    # Fallback to schoolbook
    return _build_schoolbook()
