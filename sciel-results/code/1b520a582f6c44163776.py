import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank is n*m*p = 63
    trivial_rank = n * m * p
    
    U = np.zeros((n*m, trivial_rank))
    V = np.zeros((m*p, trivial_rank))
    W = np.zeros((p*n, trivial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Try to find lower-rank decompositions
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = trivial_rank
    
    # The tensor T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # For matrix multiplication (n×m) * (m×p), the output is (n×p)
    # T[i,j,k] represents: A[i,j] * B[j,k] contributes to C[i,k]
    # So we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
    # Wait, the convention is W[k*n+i, r], so W index is k*n+i
    
    # Let's define the target tensor
    # For each (i,j,k) where i in [0,n), j in [0,m), k in [0,p):
    # The product A[i,j]*B[j,k] contributes to C[i,k]
    # The decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    target = np.zeros((n, m, p))
    target.fill(1.0)
    
    def compute_residual(U_mat, V_mat, W_mat, rank):
        """Compute the residual of the decomposition"""
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] = np.dot(U_mat[i*m+j, :rank], 
                                          V_mat[j*p+k, :rank] * W_mat[k*n+i, :rank])
        return res - target
    
    def objective(params, rank):
        """Objective function for optimization"""
        U_mat = params[0:n*m*rank].reshape(n*m, rank)
        V_mat = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
        W_mat = params[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
        
        res = compute_residual(U_mat, V_mat, W_mat, rank)
        return np.sum(res**2)
    
    # Try various ranks from low to high
    # Theoretical lower bound for matrix multiplication rank
    # For (3,3,7), we expect rank between ~15 and 63
    
    for rank in range(15, trivial_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        n_restarts = 3
        best_obj = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(42 + rank * 100 + restart)
            init_U = np.random.randn(n*m, rank)
            init_V = np.random.randn(m*p, rank)
            init_W = np.random.randn(p*n, rank)
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(objective, params, args=(rank,), method='L-BFGS-B',
                                 options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
                    
                    if best_obj < 1e-6:
                        break
                        
            except Exception:
                continue
        
        # Check if we found a valid decomposition
        if best_obj < 1e-6:
            U_mat = best_params[0:n*m*rank].reshape(n*m, rank)
            V_mat = best_params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_mat = best_params[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
            
            # Verify
            res = compute_residual(U_mat, V_mat, W_mat, rank)
            if np.max(np.abs(res)) < 1e-5:
                best_U, best_V, best_W = U_mat, V_mat, W_mat
                best_rank = rank
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
