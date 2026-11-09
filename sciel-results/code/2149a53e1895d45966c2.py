"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 25
    pn = p * n  # 15
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err <= tol
    
    def check_decomposition_fast(U, V, W, tol=1e-10):
        """Faster check using vectorization."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                v_base = j * p
                w_base = i  # W[k*n + i]
                # For each k, compute sum_c U[u_idx,c] * V[v_base+k,c] * W[w_base + k*n, c]
                for k in range(p):
                    v_idx = v_base + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err <= tol
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank 75, try to reduce
    best_result = None
    best_rank = n * m * p  # 75
    
    # First, establish the schoolbook solution as fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    # Try ranks from lower to higher
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something? 
    # Actually, the rank must be at least the "slice rank" or other lower bounds.
    # Let's try from a reasonable lower bound up to 75.
    
    # For (3,5,5), let's try ranks from 50 to 74
    # Use random restarts with alternating minimization
    
    np.random.seed(42)
    
    def optimize_rank(r, max_iters=500, tol=1e-12):
        """Try to find a rank-r decomposition using alternating least squares."""
        if time_left() < 5:
            return None
        
        # Initialize
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        prev_cost = np.inf
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # Update U: for each (i,j), we want sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c])_k = 1 for all k
            # This is: for fixed (i,j), the vector [U[i*m+j,c] for c] dotted with 
            # [V[j*p+k,c]*W[k*n+i,c] for c] should equal 1 for each k.
            # So for fixed (i,j), we have p equations: 
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1, for k=0..p-1
            # Let M_k[c] = V[j*p+k,c] * W[k*n+i,c] for each k
            # Then U[i*m+j,:] @ M_k = 1 for each k
            # Stack: M is p x r, we want U[i*m+j,:] @ M.T = ones(p)
            # So U[i*m+j,:] = ones(p) @ pinv(M.T) ... or solve least squares
            
            for i in range(n):
                for j in range(m):
                    # Build M: p x r matrix where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve: U[i*m+j, :] @ M.T = ones(p)
                    # i.e., M @ U[i*m+j, :].T = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            # For fixed (j,k): M_i[c] = U[i*m+j,c] * W[k*n+i,c]
            # Want: V[j*p+k,:] @ M_i.T = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # For fixed (k,i): M_j[c] = U[i*m+j,c] * V[j*p+k,c]
            # Want: W[k*n+i,:] @ M_j.T = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 5 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < tol:
                    return U, V, W
                if max_err < 1e-8:
                    # Do a few more iterations with tighter tolerance
                    continue
                if max_err > prev_cost * 1.01 and it > 50:
                    # Not improving, break
                    if it > 100:
                        break
                prev_cost = max_err
        
        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err < 1e-10:
            return U, V, W
        return None
    
    # Try ranks from low to high
    # Lower bound considerations: 
    # The rank must be at least max(n, m, p) for each "fiber"
    # For (3,5,5), let's try starting from 30 and going up
    for r in range(30, 75):
        if time_left() < 15:
            break
        
        found = False
        for restart in range(3):
            if time_left() < 10:
                break
            result = optimize_rank(r, max_iters=300)
            if result is not None:
                U, V, W = result
                # Verify
                if check_decomposition_fast(U, V, W, tol=1e-10):
                    best_rank = r
                    best_result = (U, V, W)
                    found = True
                    break
            if time_left() < 10:
                break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to lists
        U_list = [[float(U[i,j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[float(V[i,j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[float(W[i,j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[float(U_school[i,j]) for j in range(U_school.shape[1])] for i in range(U_school.shape[0])]
        V_list = [[float(V_school[i,j]) for j in range(V_school.shape[1])] for i in range(V_school.shape[0])]
        W_list = [[float(W_school[i,j]) for j in range(W_school.shape[1])] for i in range(W_school.shape[0])]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
