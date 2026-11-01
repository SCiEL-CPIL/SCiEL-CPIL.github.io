import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # Build the tensor T
    # T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some i,j,k
    # T[a, b, d] = 0 otherwise
    
    # We'll work with the 48 non-zero entries
    # For each (i,j,k), we have a = i*m+j, b = j*p+k, d = k*n+i
    
    # For a given rank r, we want to find U (nm x r), V (mp x r), W (pn x r)
    # such that sum_t U[a,t]*V[b,t]*W[d,t] = T[a,b,d] for all a,b,d
    
    # Strategy: alternate between solving for W (given U,V), U (given V,W), V (given U,W)
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    best_rank = n*m*p  # 48
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U0 = np.zeros((nm, best_rank))
    V0 = np.zeros((mp, best_rank))
    W0 = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank
    for r in range(n*m*p - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # Alternating minimization
            for iteration in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W given U, V
                # For each d, we need: sum_t W[d,t] * (sum over a,b of U[a,t]*V[b,t]*T[a,b,d]) = sum over a,b of T[a,b,d]
                # Wait, this isn't quite right. Let me think again.
                
                # The equation is: for all (a,b,d), sum_t U[a,t]*V[b,t]*W[d,t] = T[a,b,d]
                # If we fix U and V, this is linear in W.
                # For each d, the equations are: sum_t W[d,t] * U[a,t]*V[b,t] = T[a,b,d] for all (a,b)
                # This is a linear system: (U * V^T) * W[d,:]^T = T[:, :, d]
                # where (U*V^T)[a,b] = sum_t U[a,t]*V[b,t]
                
                # So we can compute M = U @ V.T (nm x mp)
                # Then for each d, solve M @ W[d,:]^T = T[:, :, d]
                
                M = U @ V.T  # nm x mp
                
                # T[:, :, d] is a vector of length nm*mp
                # We need to solve M @ x = T_flat for x = W[d,:]
                
                success = True
                W_new = np.zeros((pn, r))
                for d in range(pn):
                    # Get T[:, :, d]
                    T_d = np.zeros(nm * mp)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j
                                b = j*p+k
                                d_idx = k*n+i
                                if d_idx == d:
                                    T_d[a*mp+b] = 1
                    
                    # Solve M @ x = T_d
                    # M is nm x mp, T_d is nm*mp, x is mp
                    # This is an overdetermined system (nm*mp equations, mp unknowns)
                    # Use least squares
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, T_d, rcond=None)
                        W_new[d, :] = x
                    except:
                        success = False
                        break
                
                if not success:
                    break
                
                W = W_new
                
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            d = k*n+i
                            val = np.sum(U[a,:] * V[b,:] * W[d,:])
                            if abs(val - 1) > 1e-6:
                                residual += 1
                
                # Also check some zero entries
                zero_residual = 0
                for _ in range(100):  # Sample some zero entries
                    a = np.random.randint(0, nm)
                    b = np.random.randint(0, mp)
                    d = np.random.randint(0, pn)
                    # Check if (a,b,d) is a non-zero entry
                    is_nonzero = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i*m+j and b == j*p+k and d == k*n+i:
                                    is_nonzero = True
                    if not is_nonzero:
                        val = np.sum(U[a,:] * V[b,:] * W[d,:])
                        if abs(val) > 1e-6:
                            zero_residual += 1
                
                if residual == 0 and zero_residual == 0:
                    # Found a valid decomposition!
                    found = True
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break
                
                # Update U and V similarly
                # Solve for U given V, W
                M2 = V @ W.T  # mp x pn
                U_new = np.zeros((nm, r))
                for a in range(nm):
                    T_a = np.zeros(mp * pn)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a_idx = i*m+j
                                b = j*p+k
                                d = k*n+i
                                if a_idx == a:
                                    T_a[b*pn+d] = 1
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M2, T_a, rcond=None)
                        U_new[a, :] = x
                    except:
                        success = False
                        break
                if not success:
                    break
                U = U_new
                
                # Solve for V given U, W
                M3 = U @ W.T  # nm x pn
                V_new = np.zeros((mp, r))
                for b in range(mp):
                    T_b = np.zeros(nm * pn)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j
                                b_idx = j*p+k
                                d = k*n+i
                                if b_idx == b:
                                    T_b[a*pn+d] = 1
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M3, T_b, rcond=None)
                        V_new[b, :] = x
                    except:
                        success = False
                        break
                if not success:
                    break
                V = V_new
            
            if found:
                break
        
        if found:
            break
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
