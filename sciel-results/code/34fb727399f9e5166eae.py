import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def compute_loss(U, V, W):
        try:
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            return np.sum(diff * diff)
        except:
            return 1e9

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def validate(U, V, W):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=1e-6) and 
                np.allclose(V, V_snap, atol=1e-6) and 
                np.allclose(W, W_snap, atol=1e-6)):
            return False, None, None, None
            
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=1e-6):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    # Strategy: Exploit linearity in output dimension
    # Decompose into n independent (1 x m x p) tensor problems
    # For 1x5x5, schoolbook rank is 25, but maybe lower is possible
    # Let's try to find rank-minimal for each row independently, then stack
    
    # First, try to find a low-rank decomposition for a single row (1 x m x p)
    # The tensor for row i has non-zero entries at T[i*m+j, j*p+k, k*n+i] = 1
    # This is equivalent to multiplying a 1xm vector by an mxp matrix
    
    # Let's attempt to find rank r decomposition for single row
    # The single row tensor has dimension (m, p, 1) effectively, but embedded in larger space
    
    # Actually, let's think differently. The full tensor T can be written as sum over i of T_i
    # where T_i is the tensor for row i. If we find rank r_i decomposition for each T_i,
    # we can combine them to get rank sum(r_i) for the full tensor.
    
    # For a 1x5x5 multiplication (row vector times matrix), the tensor has structure:
    # T_single[j, k, 0] = 1 if we consider the output index fixed
    # Actually the output index is k*n+i, so for fixed i, output index varies with k
    
    # Let's try a direct approach: search for low rank decompositions
    # Try ranks from 10 up to 50
    
    best_solution = None
    best_rank = n * m * p  # 50
    
    # Try small ranks first using the structure
    # The tensor T = sum_{i=0}^{n-1} T_i where T_i has support only on row i of output
    
    # For each row i, T_i[a, b, c] is non-zero only when a = i*m + j, b = j*p + k, c = k*n + i
    # This is a "diagonal" type structure
    
    # Let's try to exploit this by finding decompositions for each row separately
    # then combining them
    
    # For row i, the tensor T_i can be seen as: sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is the matrix multiplication tensor for 1xm by mxp
    
    # The rank of 1xm by mxp multiplication is at most m*p (schoolbook)
    # But it might be lower. For 1x5x5, let's search
    
    # Actually, let's just try to find a global low-rank decomposition directly
    # but with better initialization that respects the row structure
    
    # Try ranks from 25 to 49
    for target_rank in range(25, min(50, best_rank)):
        if time.time() > deadline - 20:
            break
        
        # Try multiple restarts for this rank
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            R = target_rank
            U = np.random.randn(U_dim, R) * 0.5
            V = np.random.randn(V_dim, R) * 0.5
            W = np.random.randn(W_dim, R) * 0.5
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                u = x[:U_dim*R].reshape(U_dim, R)
                v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                return compute_loss(u, v, w)
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'disp': False, 'maxls': 50})
                
                if res.fun < 1e-8:
                    u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                    v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    
                    valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        break
                        
            except Exception as e:
                pass
        
        if best_solution:
            break
    
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
