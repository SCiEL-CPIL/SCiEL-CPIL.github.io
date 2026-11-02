import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # We want to find rank R < 48
    # Try ranks 47 down to 35
    # If found, return. Else return schoolbook.
    
    for rank in range(47, 34, -1):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        result = _try_find_decomposition_sa(n, m, p, rank, max_time=2.0)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _get_schoolbook(n, m, p)

def _get_schoolbook(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_decomposition_sa(n, m, p, rank, max_time=2.0):
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    
    # We search for integer solutions U', V', W' such that sum U'V'W' = 8
    # This corresponds to U=U'/2, V=V'/2, W=W'/2
    
    # Initialize with small random integers
    # Scale: values around 1 or 2
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few restarts
    for restart in range(10):
        if time.time() - start_time > max_time:
            break
            
        U = np.random.randint(-2, 3, (nm, rank))
        V = np.random.randint(-2, 3, (mp, rank))
        W = np.random.randint(-2, 3, (pn, rank))
        
        # Reshape for einsum
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        
        for step in range(2000):
            if time.time() - start_time > max_time:
                break
                
            # Calculate current error
            # X_ijk = sum_r U_ijk_r V_jkr_r W_kir_r
            # We want X_ijk = 8
            X = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = np.sum((X - 8.0)**2)
            
            if error < 1e-9:
                # Verify exactly
                if _verify_exact(n, m, p, U, V, W):
                    # Convert back to float (half-integers)
                    U_out = (U / 2.0).tolist()
                    V_out = (V / 2.0).tolist()
                    W_out = (W / 2.0).tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
                    
            # Pick a random variable to change
            # We can change U, V, or W
            # Weight by size? Just pick uniformly from all variables
            # Total variables: nm*rank + mp*rank + pn*rank
            # Let's just pick one of the 3 matrices randomly, then an entry
            mat_idx = np.random.choice(3)
            
            if mat_idx == 0:
                idx = np.random.choice(nm)
                col = np.random.choice(rank)
                # Change U[idx, col]
                old_val = U[idx, col]
                # Try changing by -1, 0, 1 (or more)
                delta = np.random.choice([-1, 1])
                new_val = old_val + delta
                U[idx, col] = new_val
                U_r[idx // m, idx % m, col] = new_val
                
            elif mat_idx == 1:
                idx = np.random.choice(mp)
                col = np.random.choice(rank)
                old_val = V[idx, col]
                delta = np.random.choice([-1, 1])
                new_val = old_val + delta
                V[idx, col] = new_val
                V_r[idx // p, idx % p, col] = new_val
                
            else:
                idx = np.random.choice(pn)
                col = np.random.choice(rank)
                old_val = W[idx, col]
                delta = np.random.choice([-1, 1])
                new_val = old_val + delta
                W[idx, col] = new_val
                W_r[idx // n, idx % n, col] = new_val
                
            # Calculate new error
            X_new = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            new_error = np.sum((X_new - 8.0)**2)
            
            # Accept if better, or with probability exp(-(new_error - error)/T)
            if new_error < error:
                error = new_error
            else:
                prob = np.exp(-(new_error - error) / T)
                if np.random.random() < prob:
                    error = new_error
                else:
                    # Revert
                    if mat_idx == 0:
                        U[idx, col] = old_val
                        U_r[idx // m, idx % m, col] = old_val
                    elif mat_idx == 1:
                        V[idx, col] = old_val
                        V_r[idx // p, idx % p, col] = old_val
                    else:
                        W[idx, col] = old_val
                        W_r[idx // n, idx % n, col] = old_val
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
    return None

def _verify_exact(n, m, p, U, V, W):
    """Verify integer solution U', V', W' where sum U'V'W' = 8"""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                if s != 8:
                    return False
    return True
