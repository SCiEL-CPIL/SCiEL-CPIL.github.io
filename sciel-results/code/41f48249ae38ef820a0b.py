import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Precompute index mappings
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def als_decompose(U_init, V_init, W_init, R, max_iters=200, tol=1e-10):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                try:
                    U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                    U[u, :] = U_u
                except:
                    pass
                    
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                try:
                    V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                    V[v, :] = V_v
                except:
                    pass
                    
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                try:
                    W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                    W[w, :] = W_w
                except:
                    pass
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[u, r] * V[v, r] * W[w, r]
                        err += (s - 1.0) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        return None

    def try_reduce_rank(U_cur, V_cur, W_cur, target_rank, time_limit=10):
        """
        Attempts to reduce the rank of the decomposition from current to target_rank.
        Uses ALS with initialization from the current solution.
        """
        current_rank = U_cur.shape[1]
        if current_rank <= target_rank:
            return U_cur, V_cur, W_cur
        
        # We will try to reduce step by step: current -> current-1 -> ... -> target
        U_work, V_work, W_work = U_cur.copy(), V_cur.copy(), W_cur.copy()
        
        for r in range(current_rank - 1, target_rank - 1, -1):
            # Try to find a rank r solution
            # Initialize by dropping the column with smallest norm or random column
            # Better: Initialize by projecting the current solution onto a random subspace of dimension r
            # Or simply: Take the first r columns and run ALS.
            
            # Let's try a few initializations for the lower rank
            success = False
            start_time = time.time()
            
            for attempt in range(5):
                if time.time() - start_time > time_limit:
                    break
                
                # Initialization strategy:
                # 1. Drop the column with the smallest L2 norm in U, V, W
                # 2. Random subspace projection
                
                if attempt == 0:
                    # Drop smallest norm column
                    norms = np.linalg.norm(U_work, axis=0)
                    idx_drop = np.argmin(norms)
                    mask = [i for i in range(r+1) if i != idx_drop]
                    U_init = U_work[:, mask]
                    V_init = V_work[:, mask]
                    W_init = W_work[:, mask]
                else:
                    # Random projection
                    Q = np.random.randn(r+1, r)
                    U_init = U_work @ Q
                    V_init = V_work @ Q
                    W_init = W_work @ Q

                res = als_decompose(U_init, V_init, W_init, r, max_iters=100)
                if res is not None:
                    U_work, V_work, W_work = res
                    success = True
                    break
            
            if not success:
                # If we couldn't reduce to r, we fail the whole chain
                return None, None, None
            
        return U_work, V_work, W_work

    start_time = time.time()
    deadline = 140

    # Try to reduce schoolbook rank 18 to 16
    # First, try direct ALS for rank 16 with random init (fast check)
    # If that fails, use greedy reduction.
    
    # 1. Try Greedy Reduction from 18 -> 17 -> 16
    U_red, V_red, W_red = try_reduce_rank(U_school, V_school, W_school, 16, time_limit=30)
    if U_red is not None:
        return {
            "U": U_red.tolist(),
            "V": V_red.tolist(),
            "W": W_red.tolist()
        }
        
    # 2. If greedy fails, try ALS with random init for 16, 17
    for R in [16, 17]:
        if time.time() > deadline:
            break
        for restart in range(10):
            if time.time() > deadline:
                break
            np.random.seed(restart * 100 + R)
            U_init = np.random.randn(n * m, R) * 0.5
            V_init = np.random.randn(m * p, R) * 0.5
            W_init = np.random.randn(p * n, R) * 0.5
            res = als_decompose(U_init, V_init, W_init, R, max_iters=100)
            if res is not None:
                U, V, W = res
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
