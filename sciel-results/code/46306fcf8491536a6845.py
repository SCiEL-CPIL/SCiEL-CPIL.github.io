"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if this is a valid multiplication term
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find lower rank decomposition
    # The matrix multiplication tensor for (n x m) * (m x p)
    # T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} in the right positions
    
    # Actually, the tensor is: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    # This is a 3-tensor with indices (a,b,c) where a in [0,n*m), b in [0,m*p), c in [0,p*n)
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Compute the tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and T[a,b,c] = 0 otherwise (for the specific structure)
        
        # Actually, for matrix multiplication, the tensor is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # This must hold for ALL i,j,k, and the decomposition should be exact.
        
        # Let's verify by checking all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try schoolbook first as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank using random initialization and optimization
    # For each candidate rank, try multiple random starts
    
    for target_rank in range(1, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Define loss function
            def loss(params):
                U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                loss_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_p[i*m+j, r] * V_p[j*p+k, r] * W_p[k*n+i, r]
                            loss_val += (s - 1.0) ** 2
                return loss_val
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
                if result.fun < 1e-8:
                    # Found a valid decomposition
                    U_p = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_p = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_p = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Convert to lists
                    U_list = U_p.tolist()
                    V_list = V_p.tolist()
                    W_list = W_p.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        break
            except Exception:
                pass
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
