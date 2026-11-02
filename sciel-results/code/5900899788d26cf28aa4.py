import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor non-zero entries
    # T[(i,j), (j,k), (k,i)] = 1
    # Indices: u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                targets.append((u_idx, v_idx, w_idx))
    
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try ranks from 32 down to 28?
    # Start with 32, which is likely solvable.
    for R in range(32, 27, -1):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank R
        success = False
        for attempt in range(20): # 20 restarts per rank
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R + attempt)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Simple gradient descent on the residual
            # Residual for each target (u,v,w): sum_r U[u,r] V[v,r] W[w,r] - 1
            # We want to minimize sum of squared residuals.
            
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                # Compute residuals
                residuals = np.zeros(len(targets))
                for idx, (u, v, w) in enumerate(targets):
                    dot = np.dot(U[u, :], V[v, :] * W[w, :])
                    residuals[idx] = dot - 1.0
                
                if np.max(np.abs(residuals)) < 1e-9:
                    success = True
                    break
                
                # Compute gradients
                # d(residual)/dU[u,r] = V[v,r] * W[w,r]
                # d(residual)/dV[v,r] = U[u,r] * W[w,r]
                # d(residual)/dW[w,r] = U[u,r] * V[v,r]
                
                # Accumulate gradients
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for idx, (u, v, w) in enumerate(targets):
                    res = residuals[idx]
                    # grad_U = res * V[v] * W[w]
                    gU[u, :] += res * (V[v, :] * W[w, :])
                    gV[v, :] += res * (U[u, :] * W[w, :])
                    gW[w, :] += res * (U[u, :] * V[v, :])
                
                # Update
                lr = 0.01
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
            if success:
                break
        
        if success:
            # Convert to lists of floats
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
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
# EVOLVE-BLOCK-END
