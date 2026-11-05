import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Search from rank 32 down to 20
    for r in range(32, 19, -1):
        if time.time() > 140: break # Safety margin
        
        success = False
        # Try a few restarts
        for restart in range(5):
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Define residuals
            def residuals(params):
                U_flat = params[0:n*m*r]
                V_flat = params[n*m*r:n*m*r + m*p*r]
                W_flat = params[n*m*r + m*p*r:]
                
                U = U_flat.reshape(n*m, r)
                V = V_flat.reshape(m*p, r)
                W = W_flat.reshape(p*n, r)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute sum over c
                            # S[i', k'] = sum_c U[i,j,c] V[j,k,c] W[k',i',c]
                            # We want S[i', k'] = delta[i, i'] delta[k, k']
                            for ip in range(n):
                                for kp in range(p):
                                    target = 1.0 if (ip == i and kp == k) else 0.0
                                    s = 0.0
                                    for c in range(r):
                                        s += U[i*m+j, c] * V[j*p+k, c] * W[kp*n+ip, c]
                                    res.append(s - target)
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.least_squares(residuals, x0, method='lm', max_nfev=1000)
                if res.success and res.cost < 1e-8:
                    U_final = res.x[0:n*m*r].reshape(n*m, r)
                    V_final = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_final = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                for ip in range(n):
                                    for kp in range(p):
                                        target = 1.0 if (ip == i and kp == k) else 0.0
                                        s = np.sum(U_final[i*m+j, :] * V_final[j*p+k, :] * W_final[kp*n+ip, :])
                                        err = abs(s - target)
                                        if err > max_err: max_err = err
                                        if err > 1e-5:
                                            raise ValueError("Verification failed")
                                        
                    best_U, best_V, best_W = U_final, V_final, W_final
                    best_rank = r
                    success = True
                    break
            except Exception:
                pass
                
        if success:
            break
            
    # Convert to list of lists of numbers
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
