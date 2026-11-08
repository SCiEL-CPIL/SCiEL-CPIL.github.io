import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Schoolbook solution as fallback
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
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Search for lower rank
    # Try ranks from 47 down to 24
    for r in range(rank_sb - 1, 23, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Using random initialization
            np.random.seed(42 + r + restart)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each i, j, solve A U_col = b
                for i in range(n):
                    for j in range(m):
                        # A is p x r
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        # Solve least squares
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i * m + j, :] = sol
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # A is n x r
                        # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            V[j * p + k, :] = sol
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # A is m x r
                        # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            W[k * n + i, :] = sol
                        except:
                            pass
            
            # Check residual
            # Compute sum_c U_c V_c W_c
            # We can check a few random indices or all
            # All indices: 48
            # Residual = max | sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1 |
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness of snapped version
                max_err_snap = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U_snap[i * m + j, c] * V_snap[j * p + k, c] * W_snap[k * n + i, c]
                            err = abs(s - 1.0)
                            if err > max_err_snap:
                                max_err_snap = err
                
                if max_err_snap < 1e-9:
                    # Convert to lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # If not exact, continue search
            
    return best_result
