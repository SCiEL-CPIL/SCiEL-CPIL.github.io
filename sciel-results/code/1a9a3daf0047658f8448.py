import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Verify function
    def verify(U, V, W, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank using structured search
    # Key insight: for each (j,k), the sub-problem is 2-dimensional (over i)
    # Try ranks from 41 down
    
    np.random.seed(42)
    
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            
            # Structured initialization: group by (j,k) pairs
            # For each (j,k), we need to handle 2 i-values
            # Use a mix of structured and random
            
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Initialize with structured pattern
            col = 0
            for j in range(m):
                for k in range(p):
                    # Allocate 2 columns for this (j,k) pair if possible
                    if col + 2 <= r:
                        U[0*m+j, col] = 1
                        U[1*m+j, col+1] = 1
                        V[j*p+k, col] = 1
                        V[j*p+k, col+1] = 1
                        W[k*n+0, col] = 1
                        W[k*n+1, col+1] = 1
                        col += 2
                    elif col < r:
                        U[0*m+j, col] = 1
                        U[1*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+0, col] = 1
                        W[k*n+1, col] = 1
                        col += 1
            
            # Add random perturbations to remaining columns
            if col < r:
                for c in range(col, r):
                    U[:, c] = np.random.randn(n*m) * 0.5
                    V[:, c] = np.random.randn(m*p) * 0.5
                    W[:, c] = np.random.randn(p*n) * 0.5
            
            # Simple local refinement: try to zero out residual
            # Compute residual
            def compute_residual(U, V, W):
                R = np.zeros_like(T)
                for c in range(len(U[0])):
                    R -= np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                R += T
                return R
            
            R = compute_residual(U, V, W)
            err = np.max(np.abs(R))
            
            if err < 1e-9:
                found = True
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
            
            # Try a few refinement steps
            for it in range(5):
                if time.time() > deadline - 30:
                    break
                
                # Random perturbation
                U += np.random.randn(n*m, r) * 0.01
                V += np.random.randn(m*p, r) * 0.01
                W += np.random.randn(p*n, r) * 0.01
                
                R = compute_residual(U, V, W)
                err = np.max(np.abs(R))
                
                if err < 1e-9:
                    found = True
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break
        
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
