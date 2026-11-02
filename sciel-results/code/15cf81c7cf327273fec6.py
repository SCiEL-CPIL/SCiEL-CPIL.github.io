import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[a,b,c]
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # 1. Get Rank 28 Baseline
    best_result = _strassen_28(n, m, p)
    
    # 2. Try to find Rank 27, 26, ... using Randomized Least Squares with Rounding
    for target_rank in range(27, 20, -1):
        if time.time() - start_time > deadline:
            break
        
        result = _search_rank_ls(target_rank, T_target, nm, mp, pn, start_time, deadline)
        if result is not None:
            if _verify(result, T_target):
                best_result = result
                break
                
    return best_result

def _search_rank_ls(target_rank, T_target, nm, mp, pn, start_time, deadline):
    """
    Search for rank-target_rank decomposition using random U, V and solving for W.
    Then check if W can be rounded to integers/half-integers.
    """
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() - start_time > deadline:
            break
        
        # Random U and V with integer/half-integer values
        # We want small values to make rounding easier
        np.random.seed(attempt)
        U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(nm, target_rank))
        V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(mp, target_rank))
        
        # We need to solve for W.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each c, we have a vector T[:, :, c] of size nm * mp.
        # T[:, :, c] = sum_r U[:, r] V[:, r] W[c, r]
        # Let M_c be the matrix of size (nm*mp) x target_rank where M_c[:, r] = U[:, r] * V[:, r] (flattened)
        # Then T_flat[:, c] = M_c @ W[:, c]
        
        # Construct M for all c at once?
        # M is a 3D tensor? No, for each c, M_c is different?
        # No, U and V do not depend on c.
        # So M_c is the SAME for all c?
        # M[:, r] = flatten(U[:, r] * V[:, r])
        # Yes, M is (nm*mp) x target_rank.
        
        M = np.zeros((nm * mp, target_rank))
        for r in range(target_rank):
            M[:, r] = (U[:, r] * V[:, r]).flatten()
        
        # T_flat is (nm*mp) x pn
        T_flat = T_target.reshape(nm * mp, pn)
        
        # Solve M @ W = T_flat
        # W is target_rank x pn
        # Use least squares
        try:
            W, residuals, rank, sv = np.linalg.lstsq(M, T_flat, rcond=None)
        except:
            continue
            
        # Check if W is close to integer/half-integer
        # Round W to nearest 0.5
        W_rounded = np.round(W * 2) / 2.0
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W_rounded)
        if np.allclose(T_recon, T_target, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
            
        # If not exact, we might try to adjust U, V slightly?
        # But the constraint is integer/half-integer.
        # If W_rounded doesn't work, this attempt fails.
        
    return None

def _strassen_28(n, m, p):
    # ... (keep existing)
    pass

def _verify(result, T_target):
    # ... (keep existing)
    pass
