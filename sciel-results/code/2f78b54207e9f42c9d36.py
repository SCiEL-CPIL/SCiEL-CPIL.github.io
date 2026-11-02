"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    T_target = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u = np.array(U[:, c]).reshape(n, m)
        v = np.array(V[:, c]).reshape(m, p)
        w = np.array(W[:, c]).reshape(p, n)
        T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _snap_to_half_integers(arr, tol=1e-3):
    """Snap values to the nearest integer or half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) > tol:
        return None
    return snapped

def _compute_error(U, V, W, n, m, p):
    """Compute the Frobenius norm of the error."""
    rank = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.linalg.norm(T_recon - np.ones((n, m, p)))

def _local_search(U, V, W, n, m, p, max_iters=1000, lr=0.01):
    """Perform a simple gradient descent to minimize the error."""
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for it in range(max_iters):
        # Compute gradients (approximate)
        # This is a simplification; a true gradient would be complex.
        # We use a random perturbation approach instead.
        pass
    
    # Use a simpler approach: random perturbation with acceptance
    best_error = _compute_error(U_cur, V_cur, W_cur, n, m, p)
    for it in range(max_iters):
        comp = np.random.randint(3)
        arr = [U_cur, V_cur, W_cur][comp]
        idx = np.random.randint(arr.shape[0])
        jdx = np.random.randint(arr.shape[1])
        old_val = arr[idx, jdx]
        new_val = old_val + np.random.randn() * lr
        arr[idx, jdx] = new_val
        
        new_error = _compute_error(U_cur, V_cur, W_cur, n, m, p)
        if new_error < best_error:
            best_error = new_error
        else:
            arr[idx, jdx] = old_val
            
    return U_cur, V_cur, W_cur, best_error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 5:
            break
        
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            
            rng = np.random.RandomState(42 + trial)
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Local search
            U_cur, V_cur, W_cur, error = _local_search(U_rand, V_rand, W_rand, n, m, p, max_iters=200, lr=0.1)
            
            # Snap to half-integers
            U_snapped = _snap_to_half_integers(U_cur, tol=1e-2)
            V_snapped = _snap_to_half_integers(V_cur, tol=1e-2)
            W_snapped = _snap_to_half_integers(W_cur, tol=1e-2)
            
            if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                U_list = [[U_snapped[i, c] for c in range(target_rank)] for i in range(n * m)]
                V_list = [[V_snapped[i, c] for c in range(target_rank)] for i in range(m * p)]
                W_list = [[W_snapped[i, c] for c in range(target_rank)] for i in range(p * n)]
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
