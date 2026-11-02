import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, atol=1e-9):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    # Discrete candidate values for entries
    # For small matrix multiplication, optimal decompositions often use these
    candidate_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    def try_discrete_rank(r, max_tries=500):
        if time.time() > deadline - 5:
            return None
        
        # We'll use a stochastic approach: randomly sample discrete matrices
        # and check if they decompose correctly
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(max_tries):
            if time.time() > deadline - 5:
                break
            
            # Generate random discrete matrices
            U = np.random.choice(candidate_values, size=(nm, r))
            V = np.random.choice(candidate_values, size=(mp, r))
            W = np.random.choice(candidate_values, size=(pn, r))
            
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-9:
                return U, V, W
        
        # If we found a good candidate, try to refine it
        if best_err < 0.1 and best_U is not None:
            # Local search: flip individual entries
            U, V, W = best_U, best_V, best_W
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                
                improved = False
                # Try flipping entries in U
                for idx in range(len(U)):
                    if time.time() > deadline - 3:
                        break
                    i, j = divmod(idx, r)
                    old_val = U[i, j]
                    for new_val in candidate_values:
                        if new_val == old_val:
                            continue
                        U[i, j] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err:
                            best_err = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            improved = True
                        else:
                            U[i, j] = old_val
                    if improved:
                        break
                
                if not improved:
                    # Try flipping entries in V
                    for idx in range(len(V)):
                        if time.time() > deadline - 3:
                            break
                        i, j = divmod(idx, r)
                        old_val = V[i, j]
                        for new_val in candidate_values:
                            if new_val == old_val:
                                continue
                            V[i, j] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err:
                                best_err = err
                                best_U, best_V, best_W = U.copy(), V.copy(), V.copy(), W.copy()
                                improved = True
                            else:
                                V[i, j] = old_val
                        if improved:
                            break
                
                if not improved:
                    # Try flipping entries in W
                    for idx in range(len(W)):
                        if time.time() > deadline - 3:
                            break
                        i, j = divmod(idx, r)
                        old_val = W[i, j]
                        for new_val in candidate_values:
                            if new_val == old_val:
                                continue
                            W[i, j] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err:
                                best_err = err
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                improved = True
                            else:
                                W[i, j] = old_val
                        if improved:
                            break
                
                if not improved:
                    break
            
            if best_err < 1e-9:
                return best_U, best_V, best_W
        
        return None
    
    # Try ranks from low to high with discrete search
    rank_configs = [
        (18, 300),
        (19, 300),
        (20, 400),
        (21, 500),
        (22, 600),
        (23, 800),
    ]
    
    for rank, tries in rank_configs:
        if time.time() > deadline - 10:
            break
        res = try_discrete_rank(rank, max_tries=tries)
        if res is not None:
            U, V, W = res
            if check_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    if not check_decomposition(U, V, W):
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
