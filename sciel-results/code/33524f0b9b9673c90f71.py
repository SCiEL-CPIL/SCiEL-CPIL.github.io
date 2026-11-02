import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial rank-120 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] for r_idx in range(r))
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank using random search + alternating minimization
    for target_rank in range(rank - 1, max(rank - 15, 105), -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 25:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            zero_cols = np.random.choice(rank, rank - target_rank, replace=False)
            U_try[:, zero_cols] = 0
            V_try[:, zero_cols] = 0
            W_try[:, zero_cols] = 0
            
            # Alternating minimization
            for iteration in range(30):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                for i in range(n):
                    for k in range(p):
                        A_mat = np.zeros((1, m * target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A_mat[0, j * target_rank + r] = V_try[j*p+k, r] * W_try[k*n+i, r]
                        b_vec = np.array([1.0])
                        try:
                            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            for j in range(m):
                                for r in range(target_rank):
                                    U_try[i*m+j, r] = sol[j * target_rank + r]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for i in range(n):
                        A_mat = np.zeros((1, p * target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat[0, k * target_rank + r] = U_try[i*m+j, r] * W_try[k*n+i, r]
                        b_vec = np.array([1.0])
                        try:
                            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            for k in range(p):
                                for r in range(target_rank):
                                    V_try[j*p+k, r] = sol[k * target_rank + r]
                        except:
                            pass
                        
                # Update W
                for k in range(p):
                    for j in range(m):
                        A_mat = np.zeros((1, n * target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A_mat[0, i * target_rank + r] = U_try[i*m+j, r] * V_try[j*p+k, r]
                        b_vec = np.array([1.0])
                        try:
                            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            for i in range(n):
                                for r in range(target_rank):
                                    W_try[k*n+i, r] = sol[i * target_rank + r]
                        except:
                            pass
                
                if verify(U_try, V_try, W_try):
                    U_rounded = np.round(U_try * 2) / 2
                    V_rounded = np.round(V_try * 2) / 2
                    W_rounded = np.round(W_try * 2) / 2
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
