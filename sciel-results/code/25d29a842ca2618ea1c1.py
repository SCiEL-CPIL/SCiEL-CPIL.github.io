import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    def verify(U, V, W, tol=1e-8):
        rank = len(U[0]) if U else 0
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Strassen 2x2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, use Strassen at the block level
    # This gives rank 49
    
    # But for 3x4x4, we can do better by embedding in 4x4x4 and restricting
    
    # Let me construct a 4x4x4 Strassen decomposition and restrict it
    
    # Actually, let me just try to find a valid decomposition for 3x4x4 directly
    # using a randomized approach
    
    def try_random_rank(r, n_attempts=10, seed=42):
        if time.time() > deadline - 5:
            return None
        
        for attempt in range(n_attempts):
            np.random.seed(seed + attempt)
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Simple gradient descent
            lr = 0.01
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(200):
                if time.time() > deadline - 10:
                    return None
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for i2 in range(n):
                                for k2 in range(p):
                                    s = 0.0
                                    for r_ in range(r):
                                        s += U[i*m+j, r_] * V[j*p+k, r_] * W[k2*n+i2, r_]
                                    exp = 1.0 if (i == i2 and k == k2) else 0.0
                                    err += (s - exp)**2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    break
                
                # Compute gradients (approximate)
                eps = 1e-4
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Only compute gradients for a subset for speed
                for i in range(n * m):
                    for r_ in range(r):
                        U[i, r_] += eps
                        err_p = 0.0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    for ii2 in range(n):
                                        for kk2 in range(p):
                                            s = 0.0
                                            for rr in range(r):
                                                s += U[ii*m+jj, rr] * V[jj*p+kk, rr] * W[kk2*n+ii2, rr]
                                            exp = 1.0 if (ii == ii2 and kk == kk2) else 0.0
                                            err_p += (s - exp)**2
                        U[i, r_] -= 2*eps
                        err_m = 0.0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    for ii2 in range(n):
                                        for kk2 in range(p):
                                            s = 0.0
                                            for rr in range(r):
                                                s += U[ii*m+jj, rr] * V[jj*p+kk, rr] * W[kk2*n+ii2, rr]
                                            exp = 1.0 if (ii == ii2 and kk == kk2) else 0.0
                                            err_m += (s - exp)**2
                        U[i, r_] += eps
                        dU[i, r_] = (err_p - err_m) / (2*eps)
                
                U -= lr * dU
                # Skip V and W gradients for speed in this rough attempt
            
            final_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                s = 0.0
                                for r_ in range(r):
                                    s += best_U[i*m+j, r_] * best_V[j*p+k, r_] * best_W[k2*n+i2, r_]
                                exp = 1.0 if (i == i2 and k == k2) else 0.0
                                final_err += (s - exp)**2
            
            if final_err < 1e-8:
                U_list = [[float(best_U[i, r_]) for r_ in range(r)] for i in range(n*m)]
                V_list = [[float(best_V[i, r_]) for r_ in range(r)] for i in range(m*p)]
                W_list = [[float(best_W[i, r_]) for r_ in range(r)] for i in range(p*n)]
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    best_result = None
    for r in [47, 46, 45, 44, 43, 42, 41, 40, 38, 36, 32, 28, 24, 20, 16, 14, 12]:
        if time.time() > deadline - 10:
            break
        result = try_random_rank(r)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
