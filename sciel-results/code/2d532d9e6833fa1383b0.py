import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 8]
DEADLINE = time.time() + 110

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify decomposition.
    U: list of lists (n*m, rank)
    V: list of lists (m*p, rank)
    W: list of lists (p*n, rank)
    """
    rank = len(U[0]) if n*m > 0 else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    # U indices: i*m + j -> (i, j)
    # V indices: j*p + k -> (j, k)
    # W indices: k*n + i -> (k, i)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=100):
    """
    Try to reduce rank from current to target_rank by setting some entries to zero
    and re-solving. Uses a simple local search approach.
    """
    current_rank = len(U[0])
    if target_rank >= current_rank:
        return None
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to remove (current_rank - target_rank) columns
    # We'll do this by zeroing out some entries and checking if we can still verify
    # For simplicity, try removing one column at a time
    
    for r in range(target_rank, current_rank):
        # Try to remove column r
        # Check if column r can be expressed as combination of others
        # For now, just try zeroing it out and see if verification still passes
        # This is a crude approach; a better one would solve the linear system
        
        # Create modified matrices with column r zeroed
        U_mod = U_np.copy()
        V_mod = V_np.copy()
        W_mod = W_np.copy()
        
        U_mod[:, r] = 0
        V_mod[:, r] = 0
        W_mod[:, r] = 0
        
        # Verify
        U_3d = U_mod.reshape(n, m, current_rank)
        V_3d = V_mod.reshape(m, p, current_rank)
        W_3d = W_mod.reshape(p, n, current_rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        expected = np.ones((n, m, p), dtype=float)
        
        if np.allclose(T, expected, atol=1e-9):
            # Column r was redundant! Remove it
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
    
    # Check if we achieved target rank
    if U_np.shape[1] == target_rank:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }
    
    return None

def _local_search(U, V, W, n, m, p, time_limit=10):
    """
    Local search to try to reduce rank by perturbing entries.
    """
    start_time = time.time()
    current_rank = len(U[0])
    best_rank = current_rank
    best_result = {"U": U, "V": V, "W": W}
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try various target ranks below current
    for target_rank in range(current_rank - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            # Continue trying to reduce further
            U = result["U"]
            V = result["V"]
            W = result["W"]
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    result = _schoolbook()
    
    # Verify to be safe
    if not _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    
    # Try local search to reduce rank
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        search_result = _local_search(result['U'], result['V'], result['W'], n, m, p, 
                                       time_limit=min(remaining_time - 5, 30))
        if search_result is not None:
            if _verify(search_result['U'], search_result['V'], search_result['W'], n, m, p):
                return search_result
    
    return result
