import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    if U_np.shape[1] != V_np.shape[1] or U_np.shape[1] != W_np.shape[1]:
        return False
        
    rank = U_np.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Check active entries: should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
                
    # Check inactive entries: should be 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid.add((u, v, w))
                
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                if (u, v, w) not in valid:
                    val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
    return True

def _schoolbook(n, m, p):
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

def _try_prune(U, V, W, n, m, p, deadline):
    """Try to remove columns from the decomposition while maintaining validity."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(rank):
            if time.time() > deadline - 2:
                break
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                improved = True
                break
    
    return U, V, W

def _random_search(n, m, p, target_rank, deadline, n_attempts=100):
    """Try random initialization with rounding to find a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    
    for _ in range(n_attempts):
        if time.time() > deadline - 1:
            break
            
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to prune the schoolbook solution
    U_pruned, V_pruned, W_pruned = _try_prune(U, V, W, n, m, p, deadline - 5)
    
    if U_pruned.shape[1] < U.shape[1]:
        return {"U": U_pruned.tolist(), "V": V_pruned.tolist(), "W": W_pruned.tolist()}
    
    # Try random search for lower ranks
    for r in range(35, 25, -1):
        if time.time() > deadline - 10:
            break
        result = _random_search(n, m, p, r, deadline - 5, n_attempts=50)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
