import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit to be safe (150s total)
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2

    def validate(U, V, W, atol=1e-5):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snapped matrices are close to original
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        # Compute tensor with snapped values
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    best_solution = None
    best_rank = n * m * p  # 50

    # Try ranks from 49 down to 45
    # We prioritize finding a solution if one exists.
    # Given the difficulty, we will try a few ranks with limited restarts.
    
    ranks_to_try = [49, 48, 47, 46, 45]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        R = target_rank
        found = False
        
        # Number of restarts depends on time remaining
        time_left = deadline - time.time()
        if time_left > 60:
            num_restarts = 10
        elif time_left > 30:
            num_restarts = 5
        else:
            num_restarts = 2
            
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            # Scaling is important. Schoolbook entries are 1.
            # Random init should be in a similar range.
            np.random.seed(42 + restart * 100 + R)
            
            U = np.random.randn(U_dim, R) * 0.5
            V = np.random.randn(V_dim, R) * 0.5
            W = np.random.randn(W_dim, R) * 0.5
            
            # Flatten for scipy
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                u = x[:U_dim*R].reshape(U_dim, R)
                v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                return compute_loss(u, v, w)
            
            # Use L-BFGS-B or BFGS
            # BFGS is often faster for small problems
            try:
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-5})
            except Exception:
                continue
                
            x_opt = res.x
            U_opt = x_opt[:U_dim*R].reshape(U_dim, R)
            V_opt = x_opt[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
            W_opt = x_opt[U_dim*R+V_dim*R:].reshape(W_dim, R)
            
            # Check loss
            final_loss = compute_loss(U_opt, V_opt, W_opt)
            
            if final_loss < 1e-4:
                # Try to validate with snapping
                valid, U_snap, V_snap, W_snap = validate(U_opt, V_opt, W_opt)
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    best_rank = R
                    found = True
                    break
                else:
                    # If it's very close but not snapped, maybe a local search around snapped values?
                    # For now, just accept if loss is extremely low and snap is close
                    if final_loss < 1e-8:
                         # Force snap and check again with tighter tolerance?
                         # The validate function already does this.
                         pass
        
        if found:
            break

    # If still not found, return schoolbook
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
