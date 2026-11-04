import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12
    
    # Build target tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.reshape(N * M, P)  # (108, 12)
    T_flat_U = T.reshape(M * P, N) # (144, 9)
    T_flat_V = T.reshape(N * P, M) # (108, 12)
    
    best_rank = 36
    best_U = None
    best_V = None
    best_W = None
    best_error = np.inf
    
    # Try ranks from 25 upward (known minimum for 3x3x4 is likely >23, let's try 25)
    for R in range(25, 37):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Structured initialization
            np.random.seed(1000 * R + attempt)
            
            U = np.random.randn(N, R) * 0.5
            V = np.random.randn(M, R) * 0.5
            W = np.random.randn(P, R) * 0.5
            
            # Add structured components
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = np.random.randint(R)
                        U[i * m + j, r] += 0.3
                        V[j * p + k, r] += 0.3
                        W[k * n + i, r] += 0.3
            
            # ALS iterations
            converged = False
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # 1. Solve for W
                # B[:, r] = vec(U[:, r] outer V[:, r])
                B = np.zeros((N * M, R))
                for r in range(R):
                    B[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                # T_flat = B @ W.T
                # W.T = lstsq(B, T_flat)
                W_T, _, _, _ = np.linalg.lstsq(B, T_flat, rcond=None)
                W = W_T.T
                
                # 2. Solve for U
                # C[:, r] = vec(V[:, r] outer W[:, r])
                C = np.zeros((M * P, R))
                for r in range(R):
                    C[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                # T_flat_U = C @ U.T
                U_T, _, _, _ = np.linalg.lstsq(C, T_flat_U, rcond=None)
                U = U_T.T
                
                # 3. Solve for V
                # D[:, r] = vec(U[:, r] outer W[:, r])
                D = np.zeros((N * P, R))
                for r in range(R):
                    D[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                # T_flat_V = D @ V.T
                V_T, _, _, _ = np.linalg.lstsq(D, T_flat_V, rcond=None)
                V = V_T.T
                
                # Check convergence
                # T_pred[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                T_pred = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_pred - T)
                
                if error < 1e-9:
                    converged = True
                    break
            
            if converged:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Convert to list format
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
