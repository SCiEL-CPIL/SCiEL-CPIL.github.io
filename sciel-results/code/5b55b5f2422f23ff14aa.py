import numpy as np
import time
import random
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Precompute target tensor
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_tensor[u, v, w] = 1.0

    def calculate_error(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Compute T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        T = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
        diff = T - target_tensor
        return np.sum(diff * diff)

    def discrete_local_search(rank, max_restarts=20, max_iters=5000):
        if time.time() > deadline:
            return None
            
        # Expanded value set to allow more complex decompositions
        vals = np.array([0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0, -2.0])
        vals_list = vals.tolist()
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random discrete values
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            curr_err = calculate_error(U, V, W)
            best_err = curr_err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if curr_err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            stuck = 0
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and position
                mat_type = random.choice(['U', 'V', 'W'])
                if mat_type == 'U':
                    mat = U
                    rows = n*m
                elif mat_type == 'V':
                    mat = V
                    rows = m*p
                else:
                    mat = W
                    rows = p*n
                    
                row = random.randint(0, rows - 1)
                col = random.randint(0, rank - 1)
                old_val = mat[row, col]
                
                # Try changing to a different value
                new_val = random.choice(vals_list)
                if new_val == old_val:
                    new_val = vals_list[(vals_list.index(old_val) + 1) % len(vals_list)]
                
                # Apply change
                mat[row, col] = new_val
                new_err = calculate_error(U, V, W)
                
                if new_err < curr_err:
                    curr_err = new_err
                    stuck = 0
                    if new_err < best_err:
                        best_err = new_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                        # Check for exact solution
                        if best_err < 1e-9:
                            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                else:
                    # Revert
                    mat[row, col] = old_val
                    stuck += 1
                    
                if stuck > 200:
                    break
                    
        return None

    # Search from low rank to high rank
    # Lower bound is max(nm, mp, pn) = 12.
    # Let's start from 15 to be safe but efficient.
    for rank in range(15, n*m*p + 1):
        if time.time() > deadline:
            break
        
        result = discrete_local_search(rank)
        if result is not None:
            return result
        
    # Fallback to schoolbook
    return schoolbook()
