"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 54, try to reduce
    deadline = time.time() + 140  # safety margin
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try rank 27: use structure where we separate by i
    # U[i*m+j, c] = U_i[j, c_i] where c_i is a block of size 9
    # This gives rank 3*9 = 27
    # Let's try to find a rank-27 decomposition
    # Structure: for each i, we have a 3x3x6 sub-tensor with fixed i
    # T[i,j,k] = 1, so for fixed i, it's 3x6 all-ones
    
    # Actually, let me try a different structured approach
    # Use the fact that the tensor is "separable" in a specific way
    
    # Try rank = 3 * 9 = 27 with block structure
    # U has 9 rows, V has 18 rows, W has 18 rows
    # Block by i: U rows 0-2 (i=0), 3-5 (i=1), 6-8 (i=2)
    # For each i-block, use 9 terms
    
    # Let me try a direct optimization approach
    # Initialize with random rank-r decomposition and optimize
    
    best_rank = n * m * p  # start with schoolbook
    best_result = None
    
    # First, create schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def schoolbook_result():
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    best_result = schoolbook_result()
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Use a structured ansatz: rank = 3 * r where r is rank for 1x3x6 with shared structure
    
    # Try rank 27 with a Kronecker structure
    # U[i*m+j, c] = a[i, j, c] where c = 0..26
    # V[j*p+k, c] = b[j, k, c]
    # W[k*n+i, c] = w[k, i, c]
    
    # For rank 27, let's try: c = i'*3 + j' where i', j' in 0..2
    # U[i*m+j, i'*3+j'] = delta(i,i') * something
    # This is getting complex. Let me use optimization.
    
    # Try rank 27
    target_rank = 27
    if time.time() < deadline - 30:
        try:
            np.random.seed(42)
            U_opt = np.random.randn(n*m, target_rank) * 0.1
            V_opt = np.random.randn(m*p, target_rank) * 0.1
            W_opt = np.random.randn(p*n, target_rank) * 0.1
            
            def loss(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank:]
                U_m = U_flat.reshape(n*m, target_rank)
                V_m = V_flat.reshape(m*p, target_rank)
                W_m = W_flat.reshape(p*n, target_rank)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_m[i*m+j, :], V_m[j*p+k, :] * W_m[k*n+i, :])
                            err += (s - 1.0) ** 2
                return err
            
            params0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
            res = minimize(loss, params0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.fun < 1e-10:
                U_final = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_final = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_final = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_final[i*m+j, :], V_final[j*p+k, :] * W_final[k*n+i, :])
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    best_rank = target_rank
                    best_result = {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
        except Exception:
            pass
    
    # Try even lower ranks
    for target_rank in [24, 21, 18, 15, 12]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        try:
            np.random.seed(42 + target_rank)
            U_opt = np.random.randn(n*m, target_rank) * 0.1
            V_opt = np.random.randn(m*p, target_rank) * 0.1
            W_opt = np.random.randn(p*n, target_rank) * 0.1
            
            def loss(params, r=target_rank):
                U_flat = params[:n*m*r]
                V_flat = params[n*m*r : n*m*r + m*p*r]
                W_flat = params[n*m*r + m*p*r:]
                U_m = U_flat.reshape(n*m, r)
                V_m = V_flat.reshape(m*p, r)
                W_m = W_flat.reshape(p*n, r)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_m[i*m+j, :], V_m[j*p+k, :] * W_m[k*n+i, :])
                            err += (s - 1.0) ** 2
                return err
            
            params0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
            res = minimize(loss, params0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.fun < 1e-10:
                U_final = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_final = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_final = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_final[i*m+j, :], V_final[j*p+k, :] * W_final[k*n+i, :])
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    best_rank = target_rank
                    best_result = {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
