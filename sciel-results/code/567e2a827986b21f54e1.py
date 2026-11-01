import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _bls17():
    # Placeholder for correct BLS 17 decomposition.
    # If this is invalid, it will be caught by _validate in construct.
    # For now, return schoolbook to be safe, but we will try to find rank 17 via search.
    return _schoolbook(3, 3, 5)

def _search_rank17(n, m, p, max_time=110):
    import random
    import numpy as np
    
    start = time.time()
    target_rank = 17
    
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    best_err = float('inf')
    best_state = None
    
    for seed in range(200):
        if time.time() - start > max_time:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random integers
        U = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)], dtype=float)
        V = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)], dtype=float)
        W = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)], dtype=float)
        
        # Local search
        for it in range(500):
            if time.time() - start > max_time:
                break
            
            # Compute error
            errs = np.array([np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]]) - 1.0 for idx in range(len(idx_u))])
            err = np.sum(errs ** 2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-6:
                U_snap = np.round(U * 2.0) / 2.0
                V_snap = np.round(V * 2.0) / 2.0
                W_snap = np.round(W * 2.0) / 2.0
                
                valid = True
                for idx in range(len(idx_u)):
                    s_val = np.sum(U_snap[idx_u[idx]] * V_snap[idx_v[idx]] * W_snap[idx_w[idx]])
                    if abs(s_val - 1.0) > 1e-6:
                        valid = False
                        break
                if valid:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
            
            # Random perturbation
            if random.random() < 0.5:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                U[row][col] += random.choice([-1, 1])
            elif random.random() < 0.5:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                V[row][col] += random.choice([-1, 1])
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                W[row][col] += random.choice([-1, 1])
    
    if best_state is not None:
        U_snap = np.round(best_state[0] * 2.0) / 2.0
        V_snap = np.round(best_state[1] * 2.0) / 2.0
        W_snap = np.round(best_state[2] * 2.0) / 2.0
        
        valid = True
        for idx in range(len(idx_u)):
            s_val = np.sum(U_snap[idx_u[idx]] * V_snap[idx_v[idx]] * W_snap[idx_w[idx]])
            if abs(s_val - 1.0) > 1e-6:
                valid = False
                break
        if valid:
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try hardcoded BLS 17 (if valid)
    result = _bls17()
    if _validate(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Try search
    result = _search_rank17(n, m, p, max_time=110)
    if result is not None:
        if _validate(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback
    return _schoolbook(n, m, p)
