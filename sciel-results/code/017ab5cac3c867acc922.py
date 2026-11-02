import numpy as np
import time
from scipy.optimize import least_squares
import itertools

DIMENSIONS = [2, 3, 6]

def verify(U, V, W, n, m, p, tol=1e-6):
    """
    Verify tensor decomposition:
    sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] == 1 for all i,j,k
    """
    r = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Flatten the tensor T[i,j,k] = sum_l U[i,j,l] V[j,k,l] W[k,i,l]
    # We want T[i,j,k] = 1 for all i,j,k
    
    # Efficient check using einsum or loops
    # T is a 3D array of shape (n, m, p)
    # T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = 2, 3, 6
    target_rank = 20 # Trying to beat 22
    
    # Indices
    # U shape: (n*m, r) -> (6, r)
    # V shape: (m*p, r) -> (18, r)
    # W shape: (p*n, r) -> (12, r)
    
    # We need to satisfy:
    # sum_l U[i,j,l] V[j,k,l] W[k,i,l] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's define the residual function for least_squares
    # We flatten U, V, W into a single vector x
    # x[0:6*r] = U.ravel()
    # x[6*r:6*r+18*r] = V.ravel()
    # x[6*r+18*r:6*r+18*r+12*r] = W.ravel()
    
    r = target_rank
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def residual(x):
        U = x[0:n_u*r].reshape(n_u, r)
        V = x[n_u*r:n_u*r+n_v*r].reshape(n_v, r)
        W = x[n_u*r+n_v*r:n_u*r+n_v*r+n_w*r].reshape(n_w, r)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                    res.append(s - 1.0)
        return np.array(res)
    
    # Initialize with a known valid decomposition of higher rank (e.g., rank 22 or 34) 
    # or random initialization. 
    # A good strategy is to start with a rank-22 solution and try to drive some columns to zero,
    # but since we are optimizing rank-20 directly, we need a good start.
    
    # Let's try multiple random restarts.
    # We can also try to construct a structured initial guess.
    # For 2x3x6, we can view it as 2x3x3 x 2 blocks.
    # If we have a rank-11 solution for 2x3x3, we can duplicate it to get rank-22.
    # Can we do better?
    
    # Let's try to find a rank-20 solution.
    # Known bounds: r(2,3,6) is likely 18 or 20?
    # r(2,3,3) = 11.
    # r(2,3,6) <= 2 * 11 = 22.
    # Is it possible to share computations between the two 3-column blocks?
    # The tensor is T[i,j,k]. k goes 0..5.
    # T[i,j,0..2] depends on B[:,0..2].
    # T[i,j,3..5] depends on B[:,3..5].
    # If B[:,3..5] is related to B[:,0..2], we can share.
    # But B is arbitrary. So the decomposition must hold for ALL B.
    # This means the structure of B doesn't restrict the rank unless we use block structure in the decomposition itself.
    
    # Let's just run the optimizer.
    
    best_solution = None
    best_error = float('inf')
    
    # Time check
    if time.time() > deadline:
        # Return a valid rank-22 or 34 solution as fallback
        # I'll implement a quick rank-34 fallback using the previous logic if needed, 
        # but for now, let's try to get a valid rank-20.
        pass

    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s for final check
            break
            
        # Random initialization
        np.random.seed(42 + restart)
        U_init = np.random.randn(n_u, r)
        V_init = np.random.randn(n_v, r)
        W_init = np.random.randn(n_w, r)
        
        # Scale down to help convergence?
        U_init *= 0.1
        V_init *= 0.1
        W_init *= 0.1
        
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        try:
            result = least_squares(residual, x0, method='lm', max_nfev=10000)
            err = result.cost
            if err < best_error:
                best_error = err
                best_solution = result.x
                
                # Check if it's a valid solution
                U_sol = best_solution[0:n_u*r].reshape(n_u, r)
                V_sol = best_solution[n_u*r:n_u*r+n_v*r].reshape(n_v, r)
                W_sol = best_solution[n_u*r+n_v*r:n_u*r+n_v*r+n_w*r].reshape(n_w, r)
                
                if verify(U_sol, V_sol, W_sol, n, m, p):
                    # Found a valid rank-20 solution!
                    # We can try to reduce rank further by checking for zero columns
                    # But for now, return rank 20.
                    return {
                        "U": U_sol.tolist(),
                        "V": V_sol.tolist(),
                        "W": W_sol.tolist()
                    }
        except Exception as e:
            pass

    # If rank 20 failed, try rank 21? Or fall back to a known valid one.
    # The previous program had a rank 34 solution.
    # Let's implement a simple rank 22 solution (two copies of rank 11 for 2x3x3)
    # But wait, the current program in the prompt *claims* to have a rank 34 solution 
    # but the code is messy and might not even work.
    # Let's construct a valid rank 22 solution programmatically.
    # We need a rank 11 solution for 2x3x3.
    
    # Let's try to optimize for rank 11 for 2x3x3 first?
    # No, that's too slow.
    
    # Let's use a known rank 11 decomposition for 2x3x3 if I can recall/derive one.
    # Actually, let's just run the optimizer for rank 22 on the full 2x3x6 problem.
    # It's easier to guarantee.
    
    r = 22
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def residual_22(x):
        U = x[0:n_u*r].reshape(n_u, r)
        V = x[n_u*r:n_u*r+n_v*r].reshape(n_v, r)
        W = x[n_u*r+n_v*r:n_u*r+n_v*r+n_w*r].reshape(n_w, r)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                    res.append(s - 1.0)
        return np.array(res)
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        np.random.seed(100 + restart)
        U_init = np.random.randn(n_u, r)
        V_init = np.random.randn(n_v, r)
        W_init = np.random.randn(n_w, r)
        U_init *= 0.1
        V_init *= 0.1
        W_init *= 0.1
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        try:
            result = least_squares(residual_22, x0, method='lm', max_nfev=10000)
            err = result.cost
            if err < 1e-6:
                U_sol = result.x[0:n_u*r].reshape(n_u, r)
                V_sol = result.x[n_u*r:n_u*r+n_v*r].reshape(n_v, r)
                W_sol = result.x[n_u*r+n_v*r:n_u*r+n_v*r+n_w*r].reshape(n_w, r)
                if verify(U_sol, V_sol, W_sol, n, m, p):
                    return {
                        "U": U_sol.tolist(),
                        "V": V_sol.tolist(),
                        "W": W_sol.tolist()
                    }
        except Exception:
            pass

    # Fallback: Hardcoded Rank 34 (or whatever valid one we can make quickly)
    # Let's make a valid Rank 36 (3 copies of rank 12? No, 2x3x3 is rank 11).
    # Let's just return a valid Rank 34 solution constructed from two Rank 17 blocks?
    # Actually, the simplest valid solution is the direct product: Rank = n*m*p = 36.
    # But we can do better.
    # Rank 22 is the target.
    
    # If all else fails, return a direct rank 36 solution.
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
