import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target_rank = 50  # Schoolbook fallback
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    random.seed(42)
    np.random.seed(42)

    def verify(U, V, W):
        r = U.shape[1]
        res = np.zeros((nm, mp, pn), dtype=int)
        for t in range(r):
            res += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    if res[u, v, w] != 1:
                        return False
        
        valid_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    valid_mask[u, v, w] = True
        
        if np.any(res[~valid_mask] != 0):
            return False
            
        return True

    def solve_for_W(U, V):
        r = U.shape[1]
        W_sol = np.zeros((pn, r), dtype=float)
        
        for w in range(pn):
            A_w = np.zeros((nm * mp, r), dtype=float)
            b_w = np.zeros(nm * mp, dtype=float)
            
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    b_w[idx] = target
                    for t in range(r):
                        A_w[idx, t] = U[u, t] * V[v, t]
                    idx += 1
            
            sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_sol[w, :] = sol
        
        return W_sol

    def round_to_half_integer(X):
        X2 = np.round(X * 2)
        return X2 / 2.0

    # Try ranks from 30 down to 20 first, then 35-49
    ranks_to_try = list(range(30, 19, -1)) + list(range(35, 39, -1)) + list(range(40, 49, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Multiple trials with different random seeds
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Use Gaussian random initialization for better conditioning
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            
            # Round U and V to half-integers
            U = round_to_half_integer(U)
            V = round_to_half_integer(V)
            
            # Check for zero columns (degenerate)
            if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
                continue
            
            # Solve for W
            W_float = solve_for_W(U, V)
            
            # Round W to half-integers
            W_r = round_to_half_integer(W_float)
            
            # Verify
            if verify(U, V, W_r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_r.tolist()
                }
            
            # Also try without rounding U,V (keep as float, only round W)
            # But the problem requires integer/half-integer entries
            # So we need U,V to be half-integers too
            
            # Try a different initialization: sparse
            if trial % 10 == 5:
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                # Randomly place 1s and -1s
                for t in range(r):
                    rows_u = np.random.choice(nm, size=min(3, nm), replace=False)
                    rows_v = np.random.choice(mp, size=min(3, mp), replace=False)
                    for ru in rows_u:
                        U[ru, t] = random.choice([-1, 1])
                    for rv in rows_v:
                        V[rv, t] = random.choice([-1, 1])
                
                if np.any(np.all(U == 0, axis=0)) or np.any(np.all(V == 0, axis=0)):
                    continue
                
                W_float = solve_for_W(U, V)
                W_r = round_to_half_integer(W_float)
                
                if verify(U, V, W_r):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_r.tolist()
                    }
    
    # Fallback: Schoolbook
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

# EVOLVE-BLOCK-END
