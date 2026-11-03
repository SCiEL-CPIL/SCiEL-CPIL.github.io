"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90
    
    # Schoolbook decomposition as fallback
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i*m+j][col] = 1
                V_fallback[j*p+k][col] = 1
                W_fallback[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    deadline = time.time() + 140
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from 89 down to 70
    for r in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Alternating least squares with convergence check
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for a in range(n*m):
                    A = np.array([np.outer(V[:,s], W[:,s]).flatten() for s in range(r)]).T
                    b = T[a].flatten()
                    U[a] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Update V
                for b_idx in range(m*p):
                    A = np.array([np.outer(U[:,s], W[:,s]).flatten() for s in range(r)]).T
                    b = T[:,b_idx,:].flatten()
                    V[b_idx] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Update W
                for c_idx in range(p*n):
                    A = np.array([np.outer(U[:,s], V[:,s]).flatten() for s in range(r)]).T
                    b = T[:,:,c_idx].flatten()
                    W[c_idx] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Check convergence
                recon = sum(np.outer(np.outer(U[:,s], V[:,s]), W[:,s]) for s in range(r))
                err_nonzero = np.max(np.abs(recon[T!=0] - 1.0))
                err_zero = np.max(np.abs(recon[T==0]))
                if err_nonzero < 1e-8 and err_zero < 1e-8:
                    break
            
            # Final check
            recon = sum(np.outer(np.outer(U[:,s], V[:,s]), W[:,s]) for s in range(r))
            if np.max(np.abs(recon[T!=0] - 1.0)) < 1e-6 and np.max(np.abs(recon[T==0])) < 1e-6:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
