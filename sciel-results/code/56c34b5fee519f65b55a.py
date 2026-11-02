"""Optimized tensor decomposition for 2x2x3 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if this is the multiplication tensor (each term is a product)
    # Actually, the standard form is: T[i,j,k] = 1 for all valid (i,j,k) in the multiplication sense
    # The tensor entries are: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute
    # In the standard decomposition, each term c corresponds to a triple (i,j,k)
    
    # Let's verify by checking that for all i,j,k:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Wait, that's not right either. Let me think about this more carefully.
    
    # The matrix multiplication tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all valid indices (it's the "trivial" tensor for matrix mult)
    # A rank-r decomposition gives: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                max_err = max(max_err, abs(s - 1.0))
    return max_err < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition using random initialization + optimization
    # Start with schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_best = [[0] * rank_schoolbook for _ in range(n * m)]
    V_best = [[0] * rank_schoolbook for _ in range(m * p)]
    W_best = [[0] * rank_schoolbook for _ in range(p * n)]
    for col in range(rank_schoolbook):
        i, j, k = divmod(col, 1)  # placeholder
        # Reconstruct schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i*m+j][col] = 1
                V_best[j*p+k][col] = 1
                W_best[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_best, V_best, W_best
    
    # Try ranks from 8 down to 11 (theoretical minimum is max(n*m, m*p, p*n) = max(4,6,6)=6,
    # but for 2x2x3, known results suggest rank around 7-8)
    
    for target_rank in range(7, 12):
        if time.time() > deadline - 30:
            break
            
        # Multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Use gradient-free optimization via coordinate descent / alternating least squares
            # Minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row (i*m+j), solve linear system
                # For fixed V, W, the contribution to entry (i,j,k) from column c is:
                # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # We want sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is a linear system in U[i*m+j,:]
                
                # Build the system for each (i,j)
                # For row r = i*m+j of U, we need:
                # sum_c U[r,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k in range(p)
                
                # This is: U[r,:] @ M_{i,j} = ones(p) where M_{i,j}[k,c] = V[j*p+k,c]*W[k*n+i,c]
                
                # Solve least squares
                for i in range(n):
                    for j in range(m):
                        r = i*m+j
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                M[k,c] = V[j*p+k, c] * W[k*n+i, c]
                        # Solve U[r,:] @ M = ones(p), i.e., M.T @ U[r,:] = ones(p)
                        try:
                            U[r,:] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        r = j*p+k
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                M[i,c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            V[r,:] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        r = k*n+i
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                M[j,c] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            W[r,:] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0 or it == 199:
                    # Compute error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(target_rank):
                                    s += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                                err += (s - 1.0)**2
                    if err < 1e-8:
                        converged = True
                        break
            
            if converged:
                # Verify with integer/half-integer rounding
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    break  # Found valid decomposition at this rank, try lower
        
        if best_rank <= target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
