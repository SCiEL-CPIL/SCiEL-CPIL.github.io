import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # The matrix multiplication tensor for 3x4 * 4x5 -> 3x5
    # We need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all other (i, j, j', k, i') combinations that don't match the pattern,
    #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r] = 0
    # Actually the full tensor T has entries T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if j == j' and k == k' and i == i', else 0.
    # But more precisely, the matmul tensor is:
    # T[i,j, j',k, k',i'] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # which is 1 only when all three deltas are 1.
    
    # Schoolbook rank = n*m*p = 60
    # We'll try to find a lower rank decomposition.
    
    # Strategy: Use the known lower bounds and known constructions.
    # For 3x4x5, the best known upper bound is 38 (from Landsberg et al.)
    # but that's complex. Let's try a systematic search with ALS.
    
    # Actually, let me think about this differently.
    # The tensor is: sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # We want to express this as sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # Let's use ALS (Alternating Least Squares) to find a rank-R decomposition.
    # We'll search R from 30 up to 59.
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matmul tensor."""
        n, m, p = DIMENSIONS
        # Check all non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check all zero entries (sample a few for speed, but for correctness check all)
        # There are (n*m)*(m*p)*(p*n) - n*m*p entries that should be 0
        # For speed, we'll check all of them since n,m,p are small
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                if not (j == jp and k == kp and i == ip):
                                    idx_a = i*m + j
                                    idx_b = jp*p + k
                                    idx_c = kp*n + ip
                                    val = np.dot(U[idx_a], V[idx_b] * W[idx_c])
                                    if abs(val) > 1e-9:
                                        return False
        return True
    
    # Try rank from 30 to 59
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    np.random.seed(42)
    
    for r in range(30, 60):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            converged = False
            for it in range(200):
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Minimize sum_{k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
                        # + sum over all (jp, kp, ip) where not (jp=j, kp=k, ip=i) of
                        #   (sum_r U[i*m+j,r] V[jp*p+k,r] W[kp*n+ip,r])^2
                        # This is a quadratic in U[i*m+j, :]
                        # Let x = U[i*m+j, :]
                        # Objective: sum_k (x . (V[j*p+k,:] * W[k*n+i,:]) - 1)^2
                        #           + sum_{jp,kp,ip not matching} (x . (V[jp*p+k,:] * W[kp*n+ip,:]))^2
                        # = x^T A x - 2 b^T x + c
                        # where A = sum_k (V[j*p+k,:]*W[k*n+i,:])^T (V[j*p+k,:]*W[k*n+i,:])
                        #         + sum_{jp,kp,ip} (V[jp*p+k,:]*W[kp*n+ip,:])^T (V[jp*p+k,:]*W[kp*n+ip,:])
                        #   - sum_k (V[j*p+k,:]*W[k*n+i,:])^T (V[j*p+k,:]*W[k*n+i,:])  [to avoid double counting]
                        # Actually, let's just build the full system.
                        
                        # For each term (jp, kp, ip), define m_{jp,kp,ip} = V[jp*p+k,:] * W[kp*n+ip,:]
                        # The contribution to the Hessian is m^T m for all terms,
                        # and the linear term is -m for terms where (jp,kp,ip) = (j,k,i)
                        
                        H = np.zeros((r, r))
                        g = np.zeros(r)
                        for jp in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    mv = V[jp*p+k] * W[kp*n+ip]
                                    H += np.outer(mv, mv)
                                    if jp == j and kp == k and ip == i:
                                        g -= mv
                        
                        # Solve H x = g
                        try:
                            x = np.linalg.solve(H + 1e-10*np.eye(r), g)
                        except np.linalg.LinAlgError:
                            x = np.linalg.lstsq(H + 1e-10*np.eye(r), g, rcond=None)[0]
                        U[i*m+j, :] = x
                
                # Update V
                for jp in range(m):
                    for kp in range(p):
                        H = np.zeros((r, r))
                        g = np.zeros(r)
                        for i in range(n):
                            for j in range(m):
                                for ip in range(n):
                                    mv = U[i*m+j] * W[kp*n+ip]
                                    H += np.outer(mv, mv)
                                    if i == jp and j == kp and ip == i:
                                        g -= mv
                        try:
                            x = np.linalg.solve(H + 1e-10*np.eye(r), g)
                        except np.linalg.LinAlgError:
                            x = np.linalg.lstsq(H + 1e-10*np.eye(r), g, rcond=None)[0]
                        V[jp*p+kp, :] = x
                
                # Update W
                for kp in range(p):
                    for ip in range(n):
                        H = np.zeros((r, r))
                        g = np.zeros(r)
                        for i in range(n):
                            for j in range(m):
                                for jp in range(m):
                                    mv = U[i*m+j] * V[jp*p+kp]
                                    H += np.outer(mv, mv)
                                    if i == ip and j == jp and kp == kp:
                                        g -= mv
                        try:
                            x = np.linalg.solve(H + 1e-10*np.eye(r), g)
                        except np.linalg.LinAlgError:
                            x = np.linalg.lstsq(H + 1e-10*np.eye(r), g, rcond=None)[0]
                        W[kp*n+ip, :] = x
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err += (val - 1.0)**2
                # Also check some zero entries
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    if not (j == jp and kp == k and i == ip):
                                        val = np.dot(U[i*m+j], V[jp*p+kp] * W[kp*n+ip])
                                        err += val**2
                
                if err < 1e-10:
                    converged = True
                    break
                
                if time.time() > deadline:
                    break
            
            if converged:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    best_rank = r
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
                elif verify(U, V, W):
                    # Use unrounded if it's valid
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            break
    
    if best_U is None:
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
