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
    target_rank = 45
    
    # Build the target tensor T
    # U dims: n*m = 10
    # V dims: m*p = 25
    # W dims: p*n = 10
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    
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

    def compute_loss(params, U, V, W):
        # params is flattened U, V, W
        # U: (U_dim, R)
        # V: (V_dim, R)
        # W: (W_dim, R)
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # Use einsum: 'ar,br,cr->abc'
        try:
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            return np.sum(diff * diff)
        except:
            return 1e9

    def snap_to_half_integers(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2

    def validate(U, V, W):
        # Check if entries are integer or half-integer
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snapped matrices are close to original
        if not (np.allclose(U, U_snap, atol=1e-6) and 
                np.allclose(V, V_snap, atol=1e-6) and 
                np.allclose(W, W_snap, atol=1e-6)):
            return False, None, None, None
            
        # Compute tensor with snapped values
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=1e-6):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    best_solution = None
    best_rank = n * m * p # 50
    
    # Try to find rank 45
    # Use random restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        R = target_rank
        U = np.random.randn(U_dim, R) * 0.5
        V = np.random.randn(V_dim, R) * 0.5
        W = np.random.randn(W_dim, R) * 0.5
        
        # Initialize with some structure? 
        # Maybe start with schoolbook and drop 5 columns?
        # Let's stick to random for now, but scale appropriately
        
        # Optimize
        # Use scipy.optimize.minimize
        from scipy.optimize import minimize
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            u = x[:U_dim*R].reshape(U_dim, R)
            v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
            w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
            return compute_loss(x, u, v, w)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'disp': False})
            
            if res.success or res.fun < 1e-4:
                u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                
                valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    best_rank = R
                    break # Found rank 45, stop searching for lower? (45 is theoretical min)
                    
        except Exception as e:
            pass

    # If rank 45 not found, return schoolbook
    if best_solution:
        U, V, W = best_solution
        # Convert to lists of floats (integers or .5)
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
