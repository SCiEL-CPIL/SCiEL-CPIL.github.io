"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition
    best_rank = rank_schoolbook
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build the target tensor
    # T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if i=i', j=j', k=k' else 0
    # Reshape into a flat vector of length (n*m)*(m*p)*(p*n)
    nm, mp, pn = n*m, m*p, p*n
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1
    
    # Try rank = rank_schoolbook - 1
    target_rank = rank_schoolbook - 1
    
    if time.time() < deadline - 30:  # Need at least 30s left
        try:
            # Initialize with random values
            np.random.seed(42)
            U_init = np.random.randn(nm, target_rank) * 0.1
            V_init = np.random.randn(mp, target_rank) * 0.1
            W_init = np.random.randn(pn, target_rank) * 0.1
            
            def residuals(x):
                U = x[:nm*target_rank].reshape(nm, target_rank)
                V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                # Compute tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is a 3-tensor contraction
                T = np.einsum('ar,br,cr->abc', U, V, W)
                return (T - target).flatten()
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = least_squares(residuals, x0, method='trf', max_nfev=1000, xtol=1e-10, ftol=1e-10)
            
            if time.time() < deadline - 10:
                if result.cost < 1e-10:
                    U = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V = result.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W = result.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Verify
                    T = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T, target, atol=1e-8):
                        # Convert to list of lists with integer/half-integer entries
                        U_list = [[float(U[i][j]) for j in range(target_rank)] for i in range(nm)]
                        V_list = [[float(V[i][j]) for j in range(target_rank)] for i in range(mp)]
                        W_list = [[float(W[i][j]) for j in range(target_rank)] for i in range(pn)]
                        
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
