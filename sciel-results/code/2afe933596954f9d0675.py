"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a_r in range(n):
        for a_c in range(m):
            for b_r in range(m):
                for b_c in range(p):
                    for c_c in range(p):
                        for c_r in range(n):
                            if a_c == b_r and c_r == a_r and c_c == b_c:
                                T[a_r*m+a_c, b_r*p+b_c, c_c*n+c_r] = 1.0
    
    def try_rank(r, max_iters=50, seed=42):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Precompute design matrices
            # For U update: A_U[:,s] = vec(V[:,s] * W[:,s])
            A_U = np.zeros((m*p * p*n, r))
            for s in range(r):
                A_U[:,s] = (V[:,s] * W[:,s]).flatten()
            
            # For V update: A_V[:,s] = vec(U[:,s] * W[:,s])
            A_V = np.zeros((n*m * p*n, r))
            for s in range(r):
                A_V[:,s] = (U[:,s] * W[:,s]).flatten()
            
            # For W update: A_W[:,s] = vec(U[:,s] * V[:,s])
            A_W = np.zeros((n*m * m*p, r))
            for s in range(r):
                A_W[:,s] = (U[:,s] * V[:,s]).flatten()
            
            # Update U
            for i in range(n*m):
                b_vec = T[i,:,:].flatten()
                U[i,:] = np.linalg.lstsq(A_U, b_vec, rcond=None)[0]
            
            # Update V
            for j in range(m*p):
                b_vec = T[:,j,:].flatten()
                V[j,:] = np.linalg.lstsq(A_V, b_vec, rcond=None)[0]
            
            # Update W
            for k in range(p*n):
                b_vec = T[:,:,k].flatten()
                W[k,:] = np.linalg.lstsq(A_W, b_vec, rcond=None)[0]
            
            # Check error
            err = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = np.dot(U[i,:], V[j,:] * W[k,:])
                        err += (val - T[i,j,k])**2
            if err < 1e-10:
                return U, V, W
        
        return None
    
    best = None
    for r in range(40, 49):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=20)
        if result is not None:
            U, V, W = result
            best = (r, U, V, W)
            break
    
    if best is None:
        r = n*m*p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best = (r, U, V, W)
    
    r, U, V, W = best
    U_list = [[float(U[i,s]) for s in range(r)] for i in range(n*m)]
    V_list = [[float(V[j,s]) for s in range(r)] for j in range(m*p)]
    W_list = [[float(W[k,s]) for s in range(r)] for k in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
