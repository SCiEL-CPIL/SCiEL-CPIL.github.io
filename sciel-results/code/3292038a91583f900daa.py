"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False
    
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True
    
    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-6):
        return False
        
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Strategy: Try ranks from 20 upward using structured initialization + optimization
    # Known bounds: R(3,3,3)=20, so R(3,3,7) >= 20
    # Upper bound: 27 (schoolbook) or potentially lower
    
    # Try different rank targets with multiple restarts
    rank_targets = [20, 21, 22, 23, 24, 25, 26, 27]
    
    for target_rank in rank_targets:
        if time.time() > deadline - 15.0:
            break
        if target_rank >= best_rank:
            break
        
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Improved structured initialization
            U = np.zeros((n*m, target_rank), dtype=np.float64)
            V = np.zeros((m*p, target_rank), dtype=np.float64)
            W = np.zeros((p*n, target_rank), dtype=np.float64)
            
            # Better initialization: use the structure of the multiplication tensor
            # Each term (i,j,k) should contribute to U[i*m+j], V[j*p+k], W[k*n+i]
            # For rank r, we need to distribute these contributions across r columns
            
            # Initialize with a pattern based on modular arithmetic
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        
                        # Distribute contributions across columns using a hash-like function
                        # This creates a more uniform spread
                        col_u = (i * p + j * 3 + k) % target_rank
                        col_v = (j * p + k * 2 + i) % target_rank
                        col_w = (k * n + i * 2 + j) % target_rank
                        
                        U[a, col_u] += 1.0
                        V[b, col_v] += 1.0
                        W[c, col_w] += 1.0
            
            # Add small random perturbation to break symmetry
            scale = 0.1
            U[:, :] += np.random.randn(n*m, target_rank) * scale
            V[:, :] += np.random.randn(m*p, target_rank) * scale
            W[:, :] += np.random.randn(p*n, target_rank) * scale
            
            try:
                from scipy.optimize import minimize
            except ImportError:
                continue
            
            def objective(x, n=n, m=m, p=p, tr=target_rank, Tt=T_target):
                U_flat = x[:n*m*tr]
                V_flat = x[n*m*tr:n*m*tr + m*p*tr]
                W_flat = x[n*m*tr + m*p*tr:]
                
                U_mat = U_flat.reshape(n*m, tr)
                V_mat = V_flat.reshape(m*p, tr)
                W_mat = W_flat.reshape(p*n, tr)
                
                T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                return np.sum((T - Tt)**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B with reasonable settings
            maxiter = 150
            if target_rank <= 22:
                maxiter = 200
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e-8:
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Try to snap to half-integers or integers
                U_list = [[float(x) for x in row] for row in U_opt]
                V_list = [[float(x) for x in row] for row in V_opt]
                W_list = [[float(x) for x in row] for row in W_opt]
                
                # Check if already valid
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found = True
                    break
                
                # Try snapping to half-integers with tighter tolerance
                def snap(val):
                    nearest = round(val * 2) / 2.0
                    if abs(val - nearest) < 1e-3:
                        return nearest
                    return val
                
                U_snap = [[snap(v) for v in row] for row in U_list]
                V_snap = [[snap(v) for v in row] for row in V_list]
                W_snap = [[snap(v) for v in row] for row in W_list]
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
            elif res.fun < 1e-4:
                # Close but not quite - try a polish pass with tighter tolerance
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                x0_polish = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
                
                def objective_polish(x, n=n, m=m, p=p, tr=target_rank, Tt=T_target):
                    U_flat = x[:n*m*tr]
                    V_flat = x[n*m*tr:n*m*tr + m*p*tr]
                    W_flat = x[n*m*tr + m*p*tr:]
                    
                    U_mat = U_flat.reshape(n*m, tr)
                    V_mat = V_flat.reshape(m*p, tr)
                    W_mat = W_flat.reshape(p*n, tr)
                    
                    T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                    return np.sum((T - Tt)**2)
                
                res2 = minimize(objective_polish, x0_polish, method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res2.fun < 1e-10:
                    U_opt = res2.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res2.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res2.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    U_list = [[float(x) for x in row] for row in U_opt]
                    V_list = [[float(x) for x in row] for row in V_opt]
                    W_list = [[float(x) for x in row] for row in W_opt]
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        found = True
                        break
                    
                    # Try snapping again
                    U_snap = [[snap(v) for v in row] for row in U_list]
                    V_snap = [[snap(v) for v in row] for row in V_list]
                    W_snap = [[snap(v) for v in row] for row in W_list]
                    
                    if _verify(U_snap, V_snap, W_snap, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found = True
                        break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
