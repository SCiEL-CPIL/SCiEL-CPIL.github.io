import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 inner, p=6 cols of B
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # M[idx1, idx2, idx3] = 1 if valid, else 0
    # idx1 = i*m+j, idx2 = j*p+k, idx3 = k*n+i
    
    # Precompute indices for fast access
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # We don't need the full M tensor if we use the structured sums.
    # M_{u,v,w} = 1 iff u=(i,j), v=(j,k), w=(k,i).
    
    best_result = None
    best_rank = None
    
    # Search ranks from 50 to 72
    # 72 is schoolbook.
    # Try to find something lower.
    
    for rank in range(50, 73):
        if time.time() > deadline - 15:
            break
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 100 + restart)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Reshape for fast contractions
            # U: (n, m, R)
            # V: (m, p, R)
            # W: (p, n, R)
            
            converged = False
            for iteration in range(1000): # More iterations possible due to speed
                if time.time() > deadline - 5:
                    break
                
                # Update U
                # N_u,c = sum_k V[j,k,c] W[k,i,c] for u=(i,j)
                # V_reshaped: (m, p, R)
                # W_reshaped: (p, n, R)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                # Sum over k (axis 1 of V_r, axis 0 of W_r)
                # Result shape: (n, m, p, R) -> sum over p -> (n, m, R)
                # N[i, j, c] = sum_k V_r[j, k, c] * W_r[k, i, c]
                N_U = np.einsum('jkc,kic->ijc', V_r, W_r)
                
                # Gram G = (V^T V) * (W^T W) element-wise? No.
                # G_c, c' = (sum_j V_j,c V_j,c') * (sum_k W_k,c W_k,c')
                # V^T V is (R, R). W^T W is (R, R).
                # G = (V^T V) * (W^T W) is NOT correct.
                # G_{c,c'} = (V^T V)_{c,c'} (W^T W)_{c,c'}?
                # Let's check:
                # G_{c,c'} = sum_{j,k} V_{j,c} W_{k,c} V_{j,c'} W_{k,c'}
                # = sum_j V_{j,c} V_{j,c'} sum_k W_{k,c} W_{k,c'}
                # = (V^T V)_{c,c'} (W^T W)_{c,c'}
                # Yes, it is element-wise multiplication of the two Gram matrices.
                
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW
                
                try:
                    # Solve G_U X^T = N_U^T
                    # X^T = G_U^{-1} N_U^T
                    # N_U is (n, m, R). Reshape to (n*m, R).
                    N_U_flat = N_U.reshape(n*m, rank)
                    U_new = np.linalg.solve(G_U, N_U_flat.T).T
                except np.linalg.LinAlgError:
                    # Regularize
                    G_U_reg = G_U + np.eye(rank) * 1e-6
                    try:
                        U_new = np.linalg.solve(G_U_reg, N_U_flat.T).T
                    except:
                        break
                
                U = U_new
                
                # Update V
                # N_v,c = sum_i U[i,j,c] W[k,i,c] for v=(j,k)
                U_r = U.reshape(n, m, rank)
                W_r = W.reshape(p, n, rank)
                
                # N[j, k, c] = sum_i U_r[i, j, c] * W_r[k, i, c]
                N_V = np.einsum('ijc,kic->jk c', U_r, W_r)
                
                UtU = U.T @ U
                WtW = W.T @ W
                G_V = UtU * WtW
                
                try:
                    N_V_flat = N_V.reshape(m*p, rank)
                    V_new = np.linalg.solve(G_V, N_V_flat.T).T
                except np.linalg.LinAlgError:
                    G_V_reg = G_V + np.eye(rank) * 1e-6
                    try:
                        V_new = np.linalg.solve(G_V_reg, N_V_flat.T).T
                    except:
                        break
                
                V = V_new
                
                # Update W
                # N_w,c = sum_j U[i,j,c] V[j,k,c] for w=(k,i)
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                
                # N[k, i, c] = sum_j U_r[i, j, c] * V_r[j, k, c]
                N_W = np.einsum('ijc,jkc->ikc', U_r, V_r)
                
                UtU = U.T @ U
                VtV = V.T @ V
                G_W = UtU * VtV
                
                try:
                    N_W_flat = N_W.reshape(p*n, rank)
                    W_new = np.linalg.solve(G_W, N_W_flat.T).T
                except np.linalg.LinAlgError:
                    G_W_reg = G_W + np.eye(rank) * 1e-6
                    try:
                        W_new = np.linalg.solve(G_W_reg, N_W_flat.T).T
                    except:
                        break
                
                W = W_new
                
                # Check convergence
                # Residual = max | sum_c U_i,c V_j,c W_k,c - M_{i,j,k} |
                # M is 1 if indices match, 0 otherwise.
                # For valid (i,j,k), sum should be 1.
                # For invalid, sum should be 0.
                # We can check a few random points or all points.
                # All points: n*m*p = 72.
                # Construct T = sum_c U[:,c] V[:,c] W[:,c]
                # But we only need to check where M is 1 or 0.
                
                # Efficient check:
                # For all i,j,k:
                # Val = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # Target = 1 if (i,j,k) is a valid triple (always true for i in 0..n, j in 0..m, k in 0..p? No.)
                # Wait, the tensor M is defined on the full index space.
                # M[idx1, idx2, idx3] = 1 if idx1, idx2, idx3 correspond to the same (i,j,k).
                # So for any i, j, k, M[i*m+j, j*p+k, k*n+i] = 1.
                # For any other combination, M = 0.
                
                # So we need:
                # 1. For all i,j,k: sum_c ... = 1
                # 2. For all "invalid" triples: sum_c ... = 0
                
                # Check 1:
                # U_r: (n,m,R), V_r: (m,p,R), W_r: (p,n,R)
                # Val[i,j,k] = sum_c U_r[i,j,c] V_r[j,k,c] W_r[k,i,c]
                Val = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                err1 = np.max(np.abs(Val - 1.0))
                
                # Check 2:
                # We need to check points where M=0.
                # There are (n*m)*(m*p)*(p*n) - n*m*p such points.
                # 12*24*18 - 72 = 5184 - 72 = 5112 points.
                # This is small enough to check fully.
                
                # Build full tensor T_full
                # T_full[i*m+j, j*p+k, k*n+i] is the only non-zero part?
                # No, the decomposition is sum_c U_v,c V_w,c W_x,c.
                # We need to compute this for all v,w,x.
                # T_full[v,w,x] = sum_c U[v,c] V[w,c] W[x,c]
                
                # This is a rank-R tensor.
                # We can compute it as:
                # T_full = sum_c U[:,c] V[:,c] W[:,c]
                # This is expensive to build fully if R is large, but R <= 72.
                # Size 12 x 24 x 18 = 5184.
                # R * 5184 operations. 72 * 5184 ~ 370k. Very fast.
                
                T_full = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    T_full += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                
                # M_full is sparse.
                # M_full[i*m+j, j*p+k, k*n+i] = 1
                # Else 0.
                
                # Create M_full
                M_full = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M_full[i*m+j, j*p+k, k*n+i] = 1.0
                
                err2 = np.max(np.abs(T_full - M_full))
                
                residual = max(err1, err2)
                
                if residual < 1e-8:
                    converged = True
                    break
            
            if converged:
                best_rank = rank
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
