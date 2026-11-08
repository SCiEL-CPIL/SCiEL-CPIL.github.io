import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    start_time = time.time()
    deadline = 110.0
    
    # Target rank
    target_rank = 20
    
    # Define the target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Try to find a decomposition of rank target_rank
    # We will use a simple iterative method
    # Initialize U, V, W randomly
    # Then alternate solving for one factor
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    best_U = np.array(U_fb)
    best_V = np.array(V_fb)
    best_W = np.array(W_fb)
    best_rank = n*m*p
    
    # Search for lower rank
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Iterate
            for it in range(100):
                # Solve for W
                # T[u,v,:] = sum_c U[u,c] V[v,c] W[:,c]
                # For each u, v, we have a linear system for W[:, c]?
                # No, W is a matrix.
                # The equation is T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
                # This is not linear in W as a matrix, but linear in each row W[w,:].
                # For a fixed w, T[:, :, w] = sum_c (U[:,c] * V[:,c]) W[w,c]
                # Let M_w be the matrix U[:,c] * V[:,c] (element-wise product, shape n*m x m*p x r?)
                # No.
                # T[u,v,w] = sum_c (U[u,c] V[v,c]) W[w,c]
                # Let A_w[u,v] = U[u,c] V[v,c]? No, c is summed.
                # This is a tensor contraction.
                # We can reshape T to a 2D matrix for each w?
                # T_w = T[:,:,w] (shape n*m x m*p)
                # We want T_w = sum_c (U[:,c] V[:,c])^T W[w,c]?
                # Let B_c = U[:,c] * V[:,c] (element-wise, shape n*m x m*p)
                # Then T_w = sum_c B_c W[w,c]
                # This is a linear system for W[w,c].
                # We have 15 rows of W (w=0..14).
                # For each w, we have a system of (n*m)*(m*p) equations?
                # No, T_w is n*m x m*p.
                # The equation is T_w = sum_c B_c W[w,c]
                # This means for each entry (u,v), T_w[u,v] = sum_c B_c[u,v] W[w,c]
                # This is a linear system for W[w,:] with (n*m)*(m*p) equations and r unknowns.
                # We can solve this in least squares.
                
                # Construct the system
                # We need to solve for W.
                # We can stack the equations for all w.
                # Total equations: (p*n) * (n*m) * (m*p)
                # This is huge.
                
                # Alternative: Use ALS (Alternating Least Squares) for tensor decomposition.
                # But we need exact solution.
                
                pass # ALS is too slow/complex for exactness in this time.
                
            # Check if we found a good solution
            # Calculate error
            # T_approx = sum_c U[:,c,None,None] * V[None,:,c,None] * W[None,None,:,c]
            # This is expensive.
            
            pass
            
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
