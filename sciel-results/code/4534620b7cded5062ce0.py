import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 110
    
    # Schoolbook fallback
    U_fb = np.zeros((20, 100))
    V_fb = np.zeros((25, 100))
    W_fb = np.zeros((20, 100))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = None
    
    # Try to find lower rank
    ranks_to_try = [99, 98, 97, 96, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    for r in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 7919 + r * 31)
            U = np.random.randn(20, r) * 0.5
            V = np.random.randn(25, r) * 0.5
            W = np.random.randn(20, r) * 0.5
            
            for it in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        u, res, rk, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        v, res, rk, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        w, res, rk, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1)**2
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
