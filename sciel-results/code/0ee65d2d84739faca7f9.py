import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build the target tensor: T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from a lower bound up to 24
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But tensor rank can be higher
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank 24
    
    # First, set up the schoolbook solution as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        # Build the residual function
        # Variables: U (6*rank), V (12*rank), W (8*rank)
        def residuals(params):
            U = params[:n*m*rank].reshape(n*m, rank)
            V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res.append(1.0 - val)
            return np.array(res)
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + rank)
            params0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.1
            # Initialize with some structure
            # Try to initialize close to a scaled version of schoolbook
            if rank >= 12:
                # Use a simple initialization
                U0 = np.random.randn(n*m, rank) * 0.5
                V0 = np.random.randn(m*p, rank) * 0.5
                W0 = np.random.randn(p*n, rank) * 0.5
                params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=5000)
                if result.cost < 1e-10:
                    # Found a solution, check if it can be snapped to half-integers
                    U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Check if snapped solution still satisfies the equations
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_snap[i*m+j, :] * V_snap[j*p+k, :] * W_snap[k*n+i, :])
                                err = abs(1.0 - val)
                                max_err = max(max_err, err)
                    
                    if max_err < 1e-9:
                        # Valid half-integer solution found
                        best_rank = rank
                        best_result = (U_snap, V_snap, W_snap)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list of lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
