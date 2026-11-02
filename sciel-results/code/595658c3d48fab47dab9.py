import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Target tensor T[i,j,k] = 1 for all i,j,k?
    # Condition: sum_r W[k,i,r] U[i,j,r] V[j,k,r] = 1 for all i,j,k.
    
    # Try ranks from 9 to 17
    for R in range(9, 18):
        if time.time() > deadline - 10:
            break
            
        # Initialize U, V, W randomly
        # Use small integers or floats
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Try a few random restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart)
            # Initialize with small random values
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                    
                # Update U
                # For each i,j: sum_r W[k,i,r] U[i,j,r] V[j,k,r] = 1
                # This is linear in U for fixed V,W?
                # sum_r U[i,j,r] (W[k,i,r] V[j,k,r]) = 1
                # Let A[i,j,r] = W[k,i,r] V[j,k,r]. But k is summed?
                # No, the equation is for EACH i,j,k.
                # sum_r U[i,j,r] W[k,i,r] V[j,k,r] = 1.
                # This is not a single linear system in U because V and W depend on k.
                # However, for fixed i,j, we have p equations (one for each k).
                # Unknowns are U[i,j,0..R-1].
                # We can solve least squares for each i,j.
                
                # Construct system for U[i,j]
                # For each k: sum_r U[i,j,r] * (W[k,i,r] * V[j,k,r]) = 1
                # Let M[i,j,k,r] = W[k,i,r] * V[j,k,r]
                # Then M[i,j,k,:] dot U[i,j,:] = 1
                
                # Similarly for V[j,k]: sum_r V[j,k,r] * (W[k,i,r] * U[i,j,r]) = 1
                # And for W[k,i]: sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1
                
                # We can iterate:
                # 1. Update U: for each i,j, solve min || M_U[i,j] U[i,j] - 1 ||^2
                # 2. Update V: for each j,k, solve min || M_V[j,k] V[j,k] - 1 ||^2
                # 3. Update W: for each k,i, solve min || M_W[k,i] W[k,i] - 1 ||^2
                
                for _ in range(1): # One sweep
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            # Matrix for U[i,j]
                            # Rows correspond to k
                            # Col r is W[k,i,r] * V[j,k,r]
                            M = np.zeros((p, R))
                            for k in range(p):
                                M[k, :] = W[k*n+i, :] * V[j*p+k, :]
                            # Solve M @ U_vec = 1
                            try:
                                U_vec, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                                U[i*m+j, :] = U_vec
                            except:
                                pass
                                
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            M = np.zeros((n, R))
                            for i in range(n):
                                M[i, :] = W[k*n+i, :] * U[i*m+j, :]
                            try:
                                V_vec, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                                V[j*p+k, :] = V_vec
                            except:
                                pass
                                
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            M = np.zeros((m, R))
                            for j in range(m):
                                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            try:
                                W_vec, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                                W[k*n+i, :] = W_vec
                            except:
                                pass
                                
                # Check error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(W[k*n+i, :], U[i*m+j, :] * V[j*p+k, :])
                            error += (val - 1)**2
                            
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if best_error < 1e-6:
                    break
                    
        if best_error < 1e-6:
            # Found a solution. Try to round to integers/half-integers
            # Scale and round
            # The values might be small or large.
            # Try to find a scaling factor
            for scale in [1, 2, 4, 0.5, 0.25]:
                U_r = np.round(best_U * scale) / scale
                V_r = np.round(best_V * scale) / scale
                W_r = np.round(best_W * scale) / scale
                
                # Verify
                error_r = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(W_r[k*n+i, :], U_r[i*m+j, :] * V_r[j*p+k, :])
                            error_r += (val - 1)**2
                            
                if error_r < 1e-6:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
