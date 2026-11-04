from scipy.optimize import minimize
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 27
    # Splitting p=5 into p1=3 and p2=2
    
    rank1 = n * m * 3  # 3 * 3 * 3 = 27? No, 9
    # Wait, n*m*p1 = 3*3*3 = 27? No, schoolbook rank is n*m*p1?
    # No, schoolbook rank for M_{n,m,p} is n*m*p.
    # For M_{3,3,3}, rank is 27? No, 3*3*3 = 27.
    # Wait, earlier I said 9. Why?
    # C_ik = sum_j A_ij B_jk.
    # Number of terms is n * m * p.
    # For 3x3x3, terms = 27.
    # My previous calculation of 9 was wrong. 9 is the number of outputs? No, 9 outputs.
    # Each output is a sum of 3 products.
    # Total products = 27.
    # So schoolbook rank for 3x3x3 is 27.
    
    # So for 3x3x5, schoolbook rank is 45.
    # Splitting into 3x3x3 (rank 27) and 3x3x2 (rank 18).
    # Total rank = 27 + 18 = 45.
    # This is the same as the full schoolbook!
    
    # So block splitting with schoolbook sub-decompositions does NOT improve the rank.
    # It just rearranges the 45 terms.
    
    # To improve, we need a sub-decomposition with rank < n*m*p_sub.
    # For M_{3,3,3}, we need rank < 27.
    # The known rank is 23.
    # If we use rank 23 for the first part, and 18 for the second, total 41.
    # 41 < 45.
    # Score = 45/41 = 1.097.
    
    # So we need a rank 23 decomposition for M_{3,3,3}.
    # Do we have one?
    # If not, we can't hardcode it.
    
    # Can we find a rank 41 decomposition via search?
    # The seed program tried to find rank 15-24.
    # We should try to find rank 41, 42, 43, 44.
    # Initialize with:
    # - 23 random terms for the first 3x3x3 block (zeros for the last 2 cols of B/C).
    # - 18 schoolbook terms for the last 3x3x2 block.
    # This gives a starting point of rank 41.
    # Then optimize.
    
    # If we can't find a rank 23 M3 decomposition, we can try to find a rank 41 decomposition of M335 directly.
    
    # Let's implement the search for rank 41.
    
    best_rank = 45
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1

    # Try ranks 44 down to 40
    for R in range(44, 39, -1):
        if time.time() > deadline - 30: break
        
        for restart in range(5):
            if time.time() > deadline - 5: break
            
            # Initialize
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Part 1: First 3 columns of B/C
            # Use random init for 23 terms? Or schoolbook 27 terms compressed?
            # Let's use random init for R terms, but bias the first R-18 terms to the first 3 cols.
            
            np.random.seed(restart + R)
            
            # Initialize V and W to be sparse in the first 3 columns for the first R-18 terms
            # This is complex. Let's just use random init and hope the optimizer finds it.
            # Or, better: Initialize with the schoolbook 45 terms, then randomly zero out 45-R terms?
            # No, that breaks the structure.
            
            # Let's try:
            # First R-18 terms: Random, but V and W have non-zeros only in first 3 cols of B/C.
            # Last 18 terms: Schoolbook for the last 2 cols.
            
            num_part1 = R - 18
            if num_part1 < 1: continue
            
            # Part 2: Schoolbook for last 2 cols
            for i in range(n):
                for j in range(m):
                    for k in range(3, 5):
                        idx = 18 + (i*m + j)*2 + (k-3) # This indexing is wrong
                        # Let's just fill the last 18 columns with schoolbook for k=3,4
                        pass
            
            # Simpler:
            # Fill last 18 columns with schoolbook for k=3,4
            col_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(3, 5):
                        # These go into the last 18 slots?
                        # No, we want them to be the last 18 terms.
                        pass
                        
            # Let's do this:
            # Terms 0 to R-19: Random, V/W non-zero only in cols 0-2.
            # Terms R-18 to R-1: Schoolbook for cols 3-4.
            
            # Initialize Part 2
            for i in range(n):
                for j in range(m):
                    for k in range(3, 5):
                        term_idx = R - 18 + (i*m + j)*2 + (k-3)
                        U[i*m+j][term_idx] = 1
                        V[j*p+k][term_idx] = 1
                        W[k*n+i][term_idx] = 1
                        
            # Initialize Part 1
            for r in range(R - 18):
                U[:, r] = np.random.randn(n*m) * 0.1
                V[:, r] = np.random.randn(m*p) * 0.1
                W[:, r] = np.random.randn(p*n) * 0.1
                # Zero out V and W for cols 3-4
                V[3*p:5*p, r] = 0 # Rows corresponding to k=3,4
                W[3*n:5*n, r] = 0 # Rows corresponding to k=3,4
                
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                            error += (val - 1.0)**2
                return error
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                
                if res.fun < 1e-6:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    is_exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                                if not np.isclose(val, 1.0, atol=1e-9):
                                    is_exact = False
                                    break
                            if not is_exact: break
                        if not is_exact: break
                        
                    if is_exact:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            except Exception:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
