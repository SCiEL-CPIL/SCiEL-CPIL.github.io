import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition as baseline
    R = n * m * p
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = R
    
    # Try to find a lower-rank decomposition using alternating minimization
    for target_rank in range(R - 1, 38, -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_mat = np.random.randn(n * m, target_rank) * 0.1
            V_mat = np.random.randn(m * p, target_rank) * 0.1
            W_mat = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            # Compute a_k = V[j*p+k, r] * W[k*n+i, r]
                            # Compute b_k = sum_{r2 != r} U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                            # Need U[i*m+j, r] * a_k + b_k = 1 for all k
                            a = np.array([V_mat[j*p+k, r] * W_mat[k*n+i, r] for k in range(p)])
                            b = np.array([sum(U_mat[i*m+j, r2] * V_mat[j*p+k, r2] * W_mat[k*n+i, r2] for r2 in range(target_rank) if r2 != r) for k in range(p)])
                            
                            # Check consistency: (1 - b_k) / a_k should be constant
                            valid = True
                            u_val = None
                            for k in range(p):
                                if abs(a[k]) < 1e-10:
                                    if abs(1.0 - b[k]) > 1e-10:
                                        valid = False
                                        break
                                else:
                                    candidate = (1.0 - b[k]) / a[k]
                                    if u_val is None:
                                        u_val = candidate
                                    elif abs(candidate - u_val) > 1e-8:
                                        valid = False
                                        break
                            
                            if valid and u_val is not None:
                                U_mat[i*m+j, r] = u_val
                            else:
                                U_mat[i*m+j, r] = 0.0
                
                # Update V (similar logic)
                for r in range(target_rank):
                    for j in range(m):
                        for k in range(p):
                            a = np.array([U_mat[i*m+j, r] * W_mat[k*n+i, r] for i in range(n)])
                            b = np.array([sum(U_mat[i*m+j, r2] * W_mat[k*n+i, r2] for r2 in range(target_rank) if r2 != r) * V_mat[j*p+k, r2] if False else 0 for i in range(n)])
                            # Simplified: just set to 0 for now
                            V_mat[j*p+k, r] = 0.0
                
                # Update W (similar logic)
                for r in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            W_mat[k*n+i, r] = 0.0
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r] for r in range(target_rank))
                            err += (val - 1.0) ** 2
                
                if err < 1e-12:
                    found = True
                    break
            
            if found:
                best_rank = target_rank
                best_U = U_mat.tolist()
                best_V = V_mat.tolist()
                best_W = W_mat.tolist()
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
