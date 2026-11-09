import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Compute tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with expected: 1 if k < 6 else 0 (for all i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _compute_error(U, V, W, target):
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can use einsum or loops. Loops are fine for small sizes if we optimize the inner part.
    # But for local search, we need fast incremental updates.
    # This function is for full recomputation when needed.
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c]
        v_c = V[:, c]
        w_c = W[:, c]
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    T[i, j, k] += u_val * v_c[j * p + k] * w_c[k * n + i]
    return np.sum((T - target) ** 2)

def _local_search(start_rank, target_tensor, max_steps=100000):
    n, m, p = DIMENSIONS
    um, vm, wm = n * m, m * p, p * n
    
    # Initialize with random half-integers
    U = np.random.choice([0, 0.5, 1, 1.5, 2], size=(um, start_rank))
    V = np.random.choice([0, 0.5, 1, 1.5, 2], size=(vm, start_rank))
    W = np.random.choice([0, 0.5, 1, 1.5, 2], size=(wm, start_rank))
    
    # Compute initial error
    # Using a helper to compute T and error
    def get_T(U, V, W):
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_c[i * m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        T[i, j, k] += u_val * v_c[j * p + k] * w_c[k * n + i]
        return T

    T_curr = get_T(U, V, W)
    current_error = np.sum((T_curr - target_tensor) ** 2)
    
    if current_error < 1e-10:
        return U, V, W, start_rank

    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    step = 0
    while time.time() - start_time < 10.0: # Limit each local search to 10s
        step += 1
        if step % 1000 == 0 and current_error < 1e-10:
            break
            
        # Pick a random matrix and entry
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            mat = U
            rows = um
            # Changing U[i*m+j, c] affects T[i,j,k] for all k
            # i = idx // m, j = idx % m
        elif mat_idx == 1:
            mat = V
            rows = vm
            # Changing V[j*p+k, c] affects T[i,j,k] for all i
            # j = idx // p, k = idx % p
        else:
            mat = W
            rows = wm
            # Changing W[k*n+i, c] affects T[i,j,k] for all j
            # k = idx // n, i = idx % n
            
        row_idx = random.randint(0, rows - 1)
        col_idx = random.randint(0, start_rank - 1)
        
        old_val = mat[row_idx, col_idx]
        # Propose new value: nearby half-integer
        # Options: old_val +/- 0.5, +/- 1.0
        # Or just random from the set
        new_val = old_val + random.choice([-1.0, -0.5, 0.5, 1.0])
        # Keep within reasonable bounds
        if new_val < -5: new_val = -5
        if new_val > 5: new_val = 5
        
        # Calculate change in error
        delta = new_val - old_val
        if delta == 0:
            continue
            
        error_change = 0.0
        if mat_idx == 0:
            i = row_idx // m
            j = row_idx % m
            for k in range(p):
                old_t = T_curr[i, j, k]
                new_t = old_t + delta * V[j * p + k, col_idx] * W[k * n + i, col_idx]
                error_change += (new_t - 1.0) ** 2 - (old_t - 1.0) ** 2
        elif mat_idx == 1:
            j = row_idx // p
            k = row_idx % p
            for i in range(n):
                old_t = T_curr[i, j, k]
                new_t = old_t + delta * U[i * m + j, col_idx] * W[k * n + i, col_idx]
                error_change += (new_t - 1.0) ** 2 - (old_t - 1.0) ** 2
        else:
            k = row_idx // n
            i = row_idx % n
            for j in range(m):
                old_t = T_curr[i, j, k]
                new_t = old_t + delta * U[i * m + j, col_idx] * V[j * p + k, col_idx]
                error_change += (new_t - 1.0) ** 2 - (old_t - 1.0) ** 2
                
        new_error = current_error + error_change
        
        if new_error < current_error:
            # Accept move
            mat[row_idx, col_idx] = new_val
            if mat_idx == 0:
                i = row_idx // m
                j = row_idx % m
                for k in range(p):
                    T_curr[i, j, k] += delta * V[j * p + k, col_idx] * W[k * n + i, col_idx]
            elif mat_idx == 1:
                j = row_idx // p
                k = row_idx % p
                for i in range(n):
                    T_curr[i, j, k] += delta * U[i * m + j, col_idx] * W[k * n + i, col_idx]
            else:
                k = row_idx // n
                i = row_idx % n
                for j in range(m):
                    T_curr[i, j, k] += delta * U[i * m + j, col_idx] * V[j * p + k, col_idx]
                    
            current_error = new_error
            if current_error < best_error:
                best_error = current_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if current_error < 1e-10:
                return U, V, W, start_rank
        else:
            # With some probability, accept worse moves (simulated annealing)
            # But for simplicity, let's just reject for now (hill climbing)
            # To escape local minima, we can occasionally accept worse moves
            if random.random() < 0.01:
                mat[row_idx, col_idx] = new_val
                if mat_idx == 0:
                    i = row_idx // m
                    j = row_idx % m
                    for k in range(p):
                        T_curr[i, j, k] += delta * V[j * p + k, col_idx] * W[k * n + i, col_idx]
                elif mat_idx == 1:
                    j = row_idx // p
                    k = row_idx % p
                    for i in range(n):
                        T_curr[i, j, k] += delta * U[i * m + j, col_idx] * W[k * n + i, col_idx]
                else:
                    k = row_idx // n
                    i = row_idx % n
                    for j in range(m):
                        T_curr[i, j, k] += delta * U[i * m + j, col_idx] * V[j * p + k, col_idx]
                current_error = new_error

    return best_U, best_V, best_W, start_rank

def construct():
    start_time = time.time()
    
    # First, return schoolbook as baseline
    best = _schoolbook()
    best_rank = 54
    
    n, m, p = DIMENSIONS
    target_tensor = np.ones((n, m, p))
    
    # Try to find a lower-rank decomposition using local search
    # Try ranks from 40 down to 20
    for target_rank in range(40, 19, -1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(3):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
                
            U, V, W, rank = _local_search(target_rank, target_tensor)
            
            # Verify
            if _verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
    return best
