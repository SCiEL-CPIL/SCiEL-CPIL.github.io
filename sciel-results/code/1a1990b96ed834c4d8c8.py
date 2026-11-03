"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """
    Verify that U @ diag @ V^T == W^T (element-wise)
    where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T = sum_c U[:,c] V[:,c] W[:,c]^T
    # T is (n*m, m*p, p*n)
    # For each (i,j,k) in tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    # Check against target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False
    
    # Check that all other entries are 0
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
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # 1. Schoolbook (rank 63) as fallback
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
    
    # Precompute target tensor T_target (n*m, m*p, p*n)
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Precompute valid mask
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_mask[a, b, c] = True
    
    from scipy.optimize import minimize
    
    def snap_to_half(val):
        nearest = round(val * 2) / 2.0
        return nearest
    
    def polish_solution(U_init, V_init, W_init, target_rank):
        """High-precision polish pass using tighter tolerances and more iterations."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_flat = x[:nm * target_rank]
            V_flat = x[nm * target_rank: nm * target_rank + mp * target_rank]
            W_flat = x[nm * target_rank + mp * target_rank:]
            
            U_mat = U_flat.reshape(nm, target_rank)
            V_mat = V_flat.reshape(mp, target_rank)
            W_mat = W_flat.reshape(pn, target_rank)
            
            T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
            return np.sum((T - T_target)**2)
        
        def gradient(x):
            U_flat = x[:nm * target_rank]
            V_flat = x[nm * target_rank: nm * target_rank + mp * target_rank]
            W_flat = x[nm * target_rank + mp * target_rank:]
            
            U_mat = U_flat.reshape(nm, target_rank)
            V_mat = V_flat.reshape(mp, target_rank)
            W_mat = W_flat.reshape(pn, target_rank)
            
            T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
            diff = 2.0 * (T - T_target)
            
            gU = np.einsum('abd,bc,cd->ad', diff, V_mat, W_mat)
            gV = np.einsum('abd,ac,cd->bd', diff, U_mat, W_mat)
            gW = np.einsum('abd,ac,bd->cd', diff, U_mat, V_mat)
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000})
            if res.fun < 1e-12:
                U_opt = res.x[:nm * target_rank].reshape(nm, target_rank)
                V_opt = res.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_opt = res.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                return U_opt, V_opt, W_opt
        except Exception:
            pass
        return None
    
    def try_rank(target_rank, max_restarts=5):
        """Try to find a valid decomposition at the given rank."""
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline - 15.0:
            return False
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10.0:
                return False
            
            # Multiple initialization strategies
            init_strategy = restart % 3
            
            if init_strategy == 0:
                # Random Gaussian
                scale = 0.5
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            elif init_strategy == 1:
                # Sparse-like initialization
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for c in range(target_rank):
                    # Each column targets a subset of indices
                    idx_u = random.randint(0, nm - 1)
                    idx_v = random.randint(0, mp - 1)
                    idx_w = random.randint(0, pn - 1)
                    U[idx_u, c] = 1.0
                    V[idx_v, c] = 1.0
                    W[idx_w, c] = 1.0
                    # Add some noise
                    U[:, c] += np.random.randn(nm) * 0.1
                    V[:, c] += np.random.randn(mp) * 0.1
                    W[:, c] += np.random.randn(pn) * 0.1
            else:
                # Structured initialization based on tensor structure
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for c in range(target_rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    if i < n and j < m and k < p:
                        U[i * m + j, c] = 1.0
                        V[j * p + k, c] = 1.0
                        W[k * n + i, c] = 1.0
                # Perturb to escape symmetry
                U += np.random.randn(nm, target_rank) * 0.05
                V += np.random.randn(mp, target_rank) * 0.05
                W += np.random.randn(pn, target_rank) * 0.05
            
            def objective(x):
                U_flat = x[:nm * target_rank]
                V_flat = x[nm * target_rank: nm * target_rank + mp * target_rank]
                W_flat = x[nm * target_rank + mp * target_rank:]
                
                U_mat = U_flat.reshape(nm, target_rank)
                V_mat = V_flat.reshape(mp, target_rank)
                W_mat = W_flat.reshape(pn, target_rank)
                
                T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                return np.sum((T - T_target)**2)
            
            def gradient(x):
                U_flat = x[:nm * target_rank]
                V_flat = x[nm * target_rank: nm * target_rank + mp * target_rank]
                W_flat = x[nm * target_rank + mp * target_rank:]
                
                U_mat = U_flat.reshape(nm, target_rank)
                V_mat = V_flat.reshape(mp, target_rank)
                W_mat = W_flat.reshape(pn, target_rank)
                
                T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                diff = 2.0 * (T - T_target)
                
                gU = np.einsum('abd,bc,cd->ad', diff, V_mat, W_mat)
                gV = np.einsum('abd,ac,cd->bd', diff, U_mat, W_mat)
                gW = np.einsum('abd,ac,bd->cd', diff, U_mat, V_mat)
                
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-11, 'maxfun': 1000})
            except Exception:
                continue
            
            # Check if we got a good solution
            if res.fun < 1e-8:
                U_opt = res.x[:nm * target_rank].reshape(nm, target_rank)
                V_opt = res.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_opt = res.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                
                # Try high-precision polish
                if time.time() < deadline - 20.0:
                    polished = polish_solution(U_opt, V_opt, W_opt, target_rank)
                    if polished is not None:
                        U_opt, V_opt, W_opt = polished
                
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_opt]
                V_list = [[float(x) for x in row] for row in V_opt]
                W_list = [[float(x) for x in row] for row in W_opt]
                
                # Try snapped version first
                U_snap = [[snap_to_half(v) for v in row] for row in U_list]
                V_snap = [[snap_to_half(v) for v in row] for row in V_list]
                W_snap = [[snap_to_half(v) for v in row] for row in W_list]
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    return True
                elif _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    return True
        
        return False
    
    # Try ranks from 50 down to 40 (known lower bound is around 40-50)
    # Use more restarts for higher ranks where solutions are more likely
    for target_rank in range(50, 39, -1):
        if time.time() > deadline - 20.0:
            break
        
        remaining_time = deadline - time.time()
        # Scale restarts based on remaining time
        if remaining_time > 60:
            max_restarts = 9
        elif remaining_time > 30:
            max_restarts = 6
        else:
            max_restarts = 4
        
        if try_rank(target_rank, max_restarts):
            # Found a valid decomposition; continue to lower ranks if time permits
            if time.time() > deadline - 25.0:
                break
    
    # Return best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
