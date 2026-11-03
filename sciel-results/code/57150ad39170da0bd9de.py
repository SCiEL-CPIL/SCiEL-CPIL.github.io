import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_solution(U, V, W):
        """Check if U, V, W form an exact decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check that other combinations give 0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        if b != c:
                            continue
                        for i in range(n):
                            for k in range(p):
                                if a == i and d == k:
                                    continue
                                s = 0
                                for r in range(rank):
                                    s += U_arr[a*m+b, r] * V_arr[c*p+d, r] * W_arr[k*n+i, r]
                                if abs(s) > 1e-9:
                                    return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return round(2*x) / 2
    
    def compute_error(U, V, W):
        """Compute the total error of the decomposition."""
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1)**2
        return err
    
    def try_rank(R, max_restarts=20, max_iters=500):
        """Try to find a decomposition of rank R using ALS."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # ALS: Fix U and V, solve for W
                W_new = np.zeros((p*n, R))
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = w
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Fix U and W, solve for V
                V_new = np.zeros((m*p, R))
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = v
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Fix V and W, solve for U
                U_new = np.zeros((n*m, R))
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = u
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                # Check convergence
                err = compute_error(U, V, W)
                if err < 1e-10:
                    break
                if abs(err - prev_err) < 1e-12 and it > 50:
                    break
                prev_err = err
            
            # Round to half-integers
            U_hi = np.round(U * 2) / 2
            V_hi = np.round(V * 2) / 2
            W_hi = np.round(W * 2) / 2
            
            # Convert to lists
            U_list = U_hi.tolist()
            V_list = V_hi.tolist()
            W_list = W_hi.tolist()
            
            # Check exactness
            if check_solution(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    for R in range(25, 101, 1):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result
    
    return schoolbook()
