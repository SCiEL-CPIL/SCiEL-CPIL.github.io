import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Function to verify a decomposition
    def verify(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Function to refine U, V, W using ALS
    def refine(U, V, W, r, max_iters=20):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W_new[k*n + i, :] = w
                    except:
                        pass
            W = W_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # For fixed j, k, we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i
                    # This is a linear system for V[j*p+k, :]
                    # Let v = V[j*p+k, :]
                    # For each i, dot(U[i*m+j, :], v * W[k*n+i, :]) = 1
                    # This is not linear in v if W is fixed?
                    # dot(u_i, v * w_i) = sum_c u_i[c] v[c] w_i[c] = sum_c v[c] (u_i[c] w_i[c])
                    # So we have a system: M @ v = ones(n)
                    # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    M_jk = np.zeros((n, r))
                    for i in range(n):
                        M_jk[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)
                        V_new[j*p + k, :] = v
                    except:
                        pass
            V = V_new
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # For fixed i, j, we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all k
                    # Let u = U[i*m+j, :]
                    # For each k, dot(u, V[j*p+k, :] * W[k*n+i, :]) = 1
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # M @ u = ones(p)
                    M_ij = np.zeros((p, r))
                    for k in range(p):
                        M_ij[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                        U_new[i*m + j, :] = u
                    except:
                        pass
            U = U_new
            
        return U, V, W

    # Search for lower rank
    # We search from low rank upwards.
    # Known lower bound for 2x2x2 is 7.
    # For 2x3x10, rank is likely around 15-20?
    # Let's start from 10 to 40.
    
    # Prioritize lower ranks, but extend to 60 to capture structural improvements (e.g., rank 55)
    # Use more restarts for lower ranks (harder to find if feasible), fewer for higher (easier if feasible)
    ranks_to_try = list(range(10, 61))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Reduce restarts for higher ranks to save time for the search
        if r < 40:
            max_restarts = 20
        elif r < 50:
            max_restarts = 10
        else:
            max_restarts = 5
            
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Scaling matters.
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Refine
            U, V, W = refine(U, V, W, r, max_iters=30)
            
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-8:
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break

    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
