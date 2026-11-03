import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin for 150s limit
    
    # Baseline: schoolbook rank
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Precompute the identity target: b[j1*m+j2] = 1 if j1==j2 else 0
    b_id = np.eye(m).flatten()
    
    # Try ranks from rank0-1 down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = 25, but realistically higher
    # We'll try a strategic set of ranks
    
    # Strategy: try ranks in descending order, with more effort on promising ranks
    # Based on literature, rank for 3x5*5x5 is likely around 60-75
    
    candidate_ranks = list(range(rank0 - 1, 55, -1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 15.0:
            break
            
        # Number of restarts: use more for lower ranks (harder but more valuable)
        n_restarts = 5 if r > 65 else 8 if r > 58 else 12
        n_iters = 80 if r > 65 else 100
        
        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
                
            np.random.seed(12345 + r * 1000 + restart * 7)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            prev_err = float('inf')
            converged = False
            
            for it in range(n_iters):
                if time.time() > deadline - 8.0:
                    break
                
                # Update W: for each (k,i), solve least squares
                for k in range(p):
                    for i in range(n):
                        # M = U[i*m:(i+1)*m, :] shape (m, r)
                        # N = V[:, k] reshaped: N[j2, r] = V[j2*p+k, r], shape (m, r)
                        M = U[i*m:(i+1)*m, :]  # (m, r)
                        N = V[np.arange(m)*p + k, :]  # (m, r)
                        # A[j1*m+j2, r] = M[j1,r] * N[j2,r]
                        A = (M[:, np.newaxis, :] * N[np.newaxis, :, :]).reshape(m*m, r)
                        w, *_ = np.linalg.lstsq(A, b_id, rcond=None)
                        W[k*n+i, :] = w
                
                # Update U: for each (i,j1), solve least squares
                for i in range(n):
                    for j1 in range(m):
                        # Equations indexed by (k, j2): A[(k*m+j2), r] = W[k*n+i,r] * V[j2*p+k,r]
                        # b[(k*m+j2)] = delta_{j1,j2}
                        A = np.zeros((p*m, r))
                        b = np.zeros(p*m)
                        for k in range(p):
                            for j2 in range(m):
                                A[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                                b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                        u, *_ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j1, :] = u
                
                # Update V: for each (j2,k), solve least squares
                for j2 in range(m):
                    for k in range(p):
                        # Equations indexed by (i, j1): A[(i*m+j1), r] = W[k*n+i,r] * U[i*m+j1,r]
                        # b[(i*m+j1)] = delta_{j1,j2}
                        A = np.zeros((n*m, r))
                        b = np.zeros(n*m)
                        for i in range(n):
                            for j1 in range(m):
                                A[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                                b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                        v, *_ = np.linalg.lstsq(A, b, rcond=None)
                        V[j2*p+k, :] = v
                
                # Compute error every 5 iterations to save time
                if it % 5 == 0 or it == n_iters - 1:
                    err = 0.0
                    for i in range(n):
                        for k in range(p):
                            for j1 in range(m):
                                for j2 in range(m):
                                    s = np.sum(W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :])
                                    target = 1.0 if j1 == j2 else 0.0
                                    err += (s - target)**2
                    
                    if err < 1e-10:
                        converged = True
                        break
                    if err > prev_err * 1.05 and it > 10:  # Diverging
                        break
                    prev_err = err
            
            # Final verification with exact check
            if converged or it == n_iters - 1:
                # Snap to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_rank = r
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                    found = True
                    break  # Found solution for this rank, try lower
        
        # If found, continue to lower ranks; if not, continue to next rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r_ in range(r):
                        s += W[k*n+i, r_] * U[i*m+j1, r_] * V[j2*p+k, r_]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > 1e-6:
                        return False
    return True
