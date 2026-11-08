import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Try to find a lower rank decomposition
    # We search for ranks from 20 down to 15
    # Time limit per rank attempt
    time_limit_per_rank = 10.0
    start_time = time.time()
    
    # Define possible values for entries
    # The problem allows integer/half-integer
    # We restrict search to small values to keep it tractable
    # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    def check_decomposition(U, V, W):
        # Compute sum u_r v_r w_r
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # T_rec: (nm, mp, pn)
        # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a tensor product sum.
        # We can compute this efficiently.
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_rec)
        return err < 1e-9

    def random_search(rank, time_limit):
        end_time = time.time() + time_limit
        while time.time() < end_time:
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            if check_decomposition(U, V, W):
                return U, V, W
        
        # Try a local search from a random start
        U = np.random.choice(vals, size=(n*m, rank))
        V = np.random.choice(vals, size=(m*p, rank))
        W = np.random.choice(vals, size=(p*n, rank))
        
        # Simple local search: perturb one entry at a time
        for _ in range(1000):
            if time.time() > end_time:
                break
            # Pick a random matrix and entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                r, c = np.random.randint(n*m), np.random.randint(rank)
                old_val = U[r, c]
                new_val = np.random.choice(vals)
                if old_val != new_val:
                    U[r, c] = new_val
                    if check_decomposition(U, V, W):
                        return U, V, W
                    U[r, c] = old_val
            elif mat_idx == 1:
                r, c = np.random.randint(m*p), np.random.randint(rank)
                old_val = V[r, c]
                new_val = np.random.choice(vals)
                if old_val != new_val:
                    V[r, c] = new_val
                    if check_decomposition(U, V, W):
                        return U, V, W
                    V[r, c] = old_val
            else:
                r, c = np.random.randint(p*n), np.random.randint(rank)
                old_val = W[r, c]
                new_val = np.random.choice(vals)
                if old_val != new_val:
                    W[r, c] = new_val
                    if check_decomposition(U, V, W):
                        return U, V, W
                    W[r, c] = old_val
                    
        return None

    # Search for lower ranks
    for rank in range(20, 14, -1):
        if time.time() > start_time + 100: # Safety margin
            break
        result = random_search(rank, time_limit_per_rank)
        if result:
            U, V, W = result
            # Convert to lists of floats (half-integers are exact in float)
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
