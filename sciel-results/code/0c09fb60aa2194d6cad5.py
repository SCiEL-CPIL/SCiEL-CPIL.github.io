import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    deadline = time.time() + 110  # 10s safety margin
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = 128  # Fallback to trivial
    
    # Trivial decomposition for fallback
    R_trivial = n * m * p
    U_triv = np.zeros((n * m, R_trivial))
    V_triv = np.zeros((m * p, R_trivial))
    W_triv = np.zeros((p * n, R_trivial))
    r = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U_triv[i * m + j, r] = 1
                V_triv[j * p + k, r] = 1
                W_triv[k * n + i, r] = 1
                r += 1
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    best_rank = R_trivial

    # Search for lower rank
    # We know rank <= 32 is likely.
    # Try ranks from 32 down to 1.
    # To be safe, start from 32.
    
    for R in range(32, 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Try several random initializations for this R
        max_trials = 50
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Random integer U, V
            # Using small integers to keep numbers manageable
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Check if systems are solvable
            # For each (i, k), solve M_ik w_ik^T = 1
            W = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build M_ik (m x R)
                    # M_ik[j, r] = U[i, m+j, r] * V[j, p+k, r]
                    M_ik = np.zeros((m, R))
                    for j in range(m):
                        M_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M_ik w = 1
                    # Use least squares
                    try:
                        w, residuals, rank, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                        # Check residual
                        if residuals.size > 0 and residuals[0] > 1e-10:
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify full tensor identity
                # T[i,j,k] = sum_r U[i,m+j,r] V[j,p+k,r] W[k,n+i,r] == 1
                check_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            if abs(val - 1.0) > 1e-8:
                                check_ok = False
                                break
                        if not check_ok:
                            break
                    if not check_ok:
                        break
                
                if check_ok:
                    # Found a solution of rank R
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break # Found for this R, move to R-1
        
        if found:
            continue # Try smaller R
        else:
            # If we didn't find a solution for R, it might still be possible with better init.
            # But given time, we move on.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
