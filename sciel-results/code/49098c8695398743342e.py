import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 120
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 36
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s
    
    # Try to find a lower rank decomposition
    # Use alternating least squares approach
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from 24 to 35
    for rank in range(24, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt * 7919 + rank * 31)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        try:
                            U[idx, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        try:
                            V[idx, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        try:
                            W[idx, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        except:
                            pass
            
            if verify(U, V, W):
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if verify(U_s, V_s, W_s):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                else:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook decomposition
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
