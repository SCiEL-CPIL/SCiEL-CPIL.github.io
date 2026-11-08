import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    # T is sparse. T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(k,i)
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    # We don't need to build the full 3D array, just the non-zero entries.
    # There are n*m*p = 50 non-zero entries.
    # Let's store them as a list of (alpha, beta, gamma)
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                targets.append((alpha, beta, gamma))
                
    # Function to check if a decomposition is exact
    def check_decomposition(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # For each target (alpha, beta, gamma), sum_r U[alpha,r] * V[beta,r] * W[gamma,r] must be 1
        # For all other (alpha, beta, gamma) combos, it must be 0?
        # Wait, the tensor T is zero elsewhere.
        # So we must check ALL alpha, beta, gamma?
        # There are nm * mp * pn = 10 * 25 * 10 = 2500 entries.
        # Only 50 are 1. The rest 2450 must be 0.
        
        # Efficient check:
        # Compute the full tensor T_calc = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # This is expensive: 2500 * R ops.
        # R=46, 2500*46 = 115k ops. Very fast.
        
        # Let's just compute the full tensor using einsum or loops.
        # T_calc[alpha, beta, gamma] = sum_r U[alpha,r] V[beta,r] W[gamma,r]
        
        # We can compute this as:
        # Temp[alpha, beta, r] = U[alpha,r] * V[beta,r]
        # T_calc[alpha, beta, gamma] = sum_r Temp[alpha, beta, r] * W[gamma,r]
        
        # Since we only need to check if it matches T, we can check the 50 targets and a few random zeros.
        # But to be safe, we should check all 2500 or use a randomized verification.
        # Given the small size, let's compute the full tensor.
        
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 10x25x10 array.
        
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check targets
        for alpha, beta, gamma in targets:
            if T_calc[alpha, beta, gamma] != 1:
                return False
                
        # Check a sample of zeros
        # To be rigorous, check all zeros.
        # Create a mask of targets
        # T_target = np.zeros((nm, mp, pn))
        # for a,b,g in targets: T_target[a,b,g] = 1
        # if not np.allclose(T_calc, T_target): return False
        
        # Construct T_target
        T_target = np.zeros((nm, mp, pn))
        for a, b, g in targets:
            T_target[a, b, g] = 1.0
            
        if not np.allclose(T_calc, T_target, atol=1e-9):
            return False
        return True

    # Search for minimal rank
    start_time = time.time()
    deadline = start_time + 100 # 100s deadline
    
    # Try ranks from 40 to 50
    # We know 50 works.
    # Try 46 first (known possible), then 45, etc.
    # If 46 fails, try 47, 48, 49.
    
    ranks_to_try = [46, 45, 44, 43, 42, 41, 40, 47, 48, 49]
    
    best_result = None
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try to find a solution of rank R
        # Use random restarts
        max_restarts = 500 if R < 45 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with integers {-1, 0, 1}
            # Sometimes halves are needed, but start with integers.
            U = np.random.randint(-1, 2, (nm, R))
            V = np.random.randint(-1, 2, (mp, R))
            W = np.random.randint(-1, 2, (pn, R))
            
            # Quick check
            if check_decomposition(U, V, W, R):
                best_result = (U, V, W, R)
                break
                
            # If not found, maybe try a local search?
            # For now, just random restarts.
            
        if best_result:
            break

    if best_result:
        U, V, W, R = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 50
        R = 50
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    U[alpha, col] = 1
                    V[beta, col] = 1
                    W[gamma, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
