"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition for n x m x p matrix multiplication."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i, j', k'), check:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'}
    # This means for fixed (i, j', k'), we need:
    # sum_j sum_k [sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i,c]] ... 
    # Actually the tensor is: T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'}
    # So: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'}
    
    # Check all combinations
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            # Compute sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
                            val = np.sum(U_np[i*m+j, :] * V_np[jp*p+k, :] * W_np[kp*n+ip, :])
                            target = 1.0 if (j == jp and k == kp) else 0.0
                            if abs(val - target) > tol:
                                return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0]*rank_schoolbook for _ in range(nm)]
    V_school = [[0]*rank_schoolbook for _ in range(mp)]
    W_school = [[0]*rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try structured decompositions
    # Strategy: Use the fact that for 3x4x5, we can try to exploit structure
    
    # Try rank from 55 down to 45
    for init_rank in range(55, 44, -1):
        if time.time() > DEADLINE - 20.0:
            break
        
        # Multiple random restarts
        for restart in range(5):
            if time.time() > DEADLINE - 20.0:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, init_rank)
            V = np.random.randn(mp, init_rank)
            W = np.random.randn(pn, init_rank)
            
            # Use gradient-free optimization (Nelder-Mead style) via scipy
            from scipy.optimize import minimize
            
            def objective(params):
                if time.time() > DEADLINE - 15.0:
                    return 1e10
                U_flat = params[:nm*init_rank].reshape(nm, init_rank)
                V_flat = params[nm*init_rank:nm*init_rank+mp*init_rank].reshape(mp, init_rank)
                W_flat = params[nm*init_rank+mp*init_rank:].reshape(pn, init_rank)
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        val = np.sum(U_flat[i*m+j, :] * V_flat[jp*p+k, :] * W_flat[kp*n+ip, :])
                                        target = 1.0 if (j == jp and k == kp) else 0.0
                                        err += (val - target) ** 2
                return err
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if result.fun < 1e-6:
                    U_opt = result.x[:nm*init_rank].reshape(nm, init_rank)
                    V_opt = result.x[nm*init_rank:nm*init_rank+mp*init_rank].reshape(mp, init_rank)
                    W_opt = result.x[nm*init_rank+mp*init_rank:].reshape(pn, init_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify rounded version
                    if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-6):
                        if init_rank < best_rank:
                            best_rank = init_rank
                            best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        break
                    elif verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-6):
                        # Use unrounded if rounded doesn't work
                        if init_rank < best_rank:
                            best_rank = init_rank
                            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        break
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
