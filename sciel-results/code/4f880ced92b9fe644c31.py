"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 100
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # We want to find rank-r factor matrices U (nm x r), V (mp x r), W (pn x r)
    # such that T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # This is equivalent to: for each c, the outer product U[:,c] * V[:,c] * W[:,c]
    # summed over c equals T.
    
    # Strategy: Use a randomized/local optimization approach to find a low-rank decomposition.
    # Start with a moderate rank and try to minimize the reconstruction error.
    
    # Known results: For 4x5 * 5x5, the schoolbook is 100.
    # We can try to find a decomposition with rank < 100.
    
    # Approach: Start with a random initialization and use gradient-based optimization
    # to minimize the Frobenius norm of the difference between T and the reconstructed tensor.
    
    # Let's try a few different rank values and pick the best one that achieves near-zero error.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, get the schoolbook as a fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    # We'll try ranks from 80 down to 50, using random restarts
    
    for rank in range(80, 49, -5):
        if time.time() > deadline - 30:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(hash((rank, attempt)) % (2**32))
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Flatten parameters
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_p = params[:nm*rank].reshape(nm, rank)
                V_p = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_p = params[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Reconstruct tensor
                # T_recon[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
                # This can be computed efficiently using einsum
                T_recon = np.einsum('ic,jc,kc->ijk', U_p, V_p, W_p)
                
                # Frobenius norm of difference
                diff = T - T_recon
                return np.sum(diff * diff)
            
            # Use L-BFGS-B for optimization
            try:
                res = minimize(objective, params, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-8:
                    U_opt = res.x[:nm*rank].reshape(nm, rank)
                    V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Verify the reconstruction
                    T_recon = np.einsum('ic,jc,kc->ijk', U_opt, V_opt, W_opt)
                    error = np.sum((T - T_recon)**2)
                    
                    if error < 1e-6:
                        # Found a valid decomposition
                        if rank < best_rank:
                            best_rank = rank
                            best_result = (U_opt, V_opt, W_opt)
                            break
            except Exception:
                pass
    
    # If we found a better result, use it; otherwise use schoolbook
    if best_result is not None:
        U_final, V_final, W_final = best_result
    else:
        U_final, V_final, W_final = U_school, V_school, W_school
    
    # Convert to lists for the required output format
    U_out = [[float(U_final[i][j]) for j in range(U_final.shape[1])] for i in range(U_final.shape[0])]
    V_out = [[float(V_final[i][j]) for j in range(V_final.shape[1])] for i in range(V_final.shape[0])]
    W_out = [[float(W_final[i][j]) for j in range(W_final.shape[1])] for i in range(W_final.shape[0])]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
