import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Build the tensor as a sparse representation
    # T[(i,j), (j',k), (k',i')] = 1 if i==i', j==j', k==k'
    # Non-zero entries: for each (i,j,k), position ((i,j), (j,k), (k,i))
    
    # For a rank-r decomposition:
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # The decomposition requires: for each (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other triples (a,b), (c,d), (e,f) where not (a==e, b==c, d==f):
    #   sum_r U[a*m+b, r] * V[c*p+d, r] * W[e*n+f, r] = 0
    
    # This is a system of equations. Let's set it up.
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Total number of constraints: nm * mp * pn (but most are zero constraints)
    # Non-zero constraints: n*m*p = 90
    # Zero constraints: nm*mp*pn - n*m*p
    
    # For optimization, we can focus on the non-zero constraints first,
    # then verify zero constraints.
    
    # Let's try alternating least squares.
    # For fixed V, W, solve for U:
    #   For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    # This is a linear system for U[i*m+j, :]
    
    def try_rank(r, max_iters=50, seed=42):
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For each k, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise), shape (r,)
                    # Then we need: U[i*m+j, :] @ A_k = 1 for each k
                    # This is: M @ U[i*m+j, :]^T = ones(p) where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.outer(V[j*p:(j+1)*p, :], W[(i):pn:n, :])  # shape (p, r)
                    # Solve M @ x = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For each i, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let B_i = U[i*m+j, :] * W[k*n+i, :] (element-wise), shape (r,)
                    # Then we need: V[j*p+k, :] @ B_i = 1 for each i
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    M = np.outer(U[:nm:n, :], W[k:pn:n, :])  # shape (n, r)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for i in range(n):
                for k in range(p):
                    # For each j, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let C_j = U[i*m+j, :] * V[j*p+k, :] (element-wise), shape (r,)
                    # Then we need: W[k*n+i, :] @ C_j = 1 for each j
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M = np.outer(U[i*m:(i+1)*m, :], V[k:mp:p, :])  # shape (m, r)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Check if the decomposition is valid
        # Check non-zero constraints
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return None
        
        # Check zero constraints (sample some)
        # For triples where not (i==i', j==j', k==k')
        # This is expensive, so let's check all
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Extract indices
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if i == i2 and j == j2 and k == k2:
                        continue  # This is a non-zero constraint, already checked
                    
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return None
        
        # Valid! Now snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify snapped version
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_snap[i*m+j, :], V_snap[j*p+k, :] * W_snap[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return None
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if i == i2 and j == j2 and k == k2:
                        continue
                    
                    val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
                    if abs(val) > 1e-9:
                        return None
        
        return U_snap, V_snap, W_snap
    
    # Try ranks from lower bound to 90
    # Lower bound: max(nm, mp, pn) = 30
    best_result = None
    
    for r in range(30, 91):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=20, seed=r)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, r)
            break  # Found the minimum rank
    
    if best_result is None:
        # Fallback to schoolbook rank 90
        r = 90
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        best_result = (U, V, W, r)
    
    U, V, W, r = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
