import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def tensor_product(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def verify(U, V, W, tol=1e-9):
        R = tensor_product(U, V, W)
        return np.allclose(R, T, atol=tol, rtol=0)
    
    def snap_to_half_integer(x, tol=1e-6):
        x2 = 2 * x
        rounded = np.round(x2)
        if np.max(np.abs(x2 - rounded)) < tol:
            return rounded / 2.0
        return None
    
    def snap_and_verify(U, V, W, tol=1e-5):
        U_snap = snap_to_half_integer(U, tol)
        V_snap = snap_to_half_integer(V, tol)
        W_snap = snap_to_half_integer(W, tol)
        if U_snap is not None and V_snap is not None and W_snap is not None:
            if verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
        return None
    
    # Schoolbook solution (rank 30)
    U_s = np.zeros((n*m, 30))
    V_s = np.zeros((m*p, 30))
    W_s = np.zeros((p*n, 30))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_rank = 30
    best_decomp = (U_s, V_s, W_s)
    
    # Precompute indices for the 30 tensor entries
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
    
    def make_objective_and_grad(r):
        """Create objective and gradient functions for rank r"""
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        def obj(p_arr):
            U_ = p_arr[0:size_U].reshape(n*m, r)
            V_ = p_arr[size_U:size_U + size_V].reshape(m*p, r)
            W_ = p_arr[size_U + size_V:].reshape(p*n, r)
            R = tensor_product(U_, V_, W_)
            return np.sum((R - T)**2)
        
        def grad(p_arr):
            U_ = p_arr[0:size_U].reshape(n*m, r)
            V_ = p_arr[size_U:size_U + size_V].reshape(m*p, r)
            W_ = p_arr[size_U + size_V:].reshape(p*n, r)
            R = tensor_product(U_, V_, W_)
            diff = R - T
            gU = 2 * np.einsum('abc,br,cr->ar', diff, V_, W_)
            gV = 2 * np.einsum('abc,ar,cr->br', diff, U_, W_)
            gW = 2 * np.einsum('abc,ar,br->cr', diff, U_, V_)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        return obj, grad
    
    def optimize_from_init(U_init, V_init, W_init, r, max_iter=500):
        """Optimize from a given initialization"""
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        obj, grad = make_objective_and_grad(r)
        
        try:
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxls': 50, 'maxfun': 10000})
            
            if res.fun < 1e-8:
                U_opt = res.x[0:size_U].reshape(n*m, r)
                V_opt = res.x[size_U:size_U + size_V].reshape(m*p, r)
                W_opt = res.x[size_U + size_V:].reshape(p*n, r)
                
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
        except Exception:
            pass
        return None
    
    # Try ranks from 20 up to 29
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy 1: Merge schoolbook columns into r groups
        # Create r groups by partitioning the 30 indices
        num_groups = r
        group_size = 30 // num_groups
        remainder = 30 % num_groups
        
        group_assignments = []
        idx = 0
        for g in range(num_groups):
            size = group_size + (1 if g < remainder else 0)
            group_assignments.append(list(range(idx, idx + size)))
            idx += size
        
        # Try different merging strategies
        for merge_strategy in range(3):
            if time.time() > deadline:
                break
            
            U_init = np.zeros((n*m, r))
            V_init = np.zeros((m*p, r))
            W_init = np.zeros((p*n, r))
            
            for g, group in enumerate(group_assignments):
                for idx_in_group in group:
                    i, j, k = indices[idx_in_group]
                    U_init[i*m+j, g] += 1.0 / len(group)
                    V_init[j*p+k, g] += 1.0 / len(group)
                    W_init[k*n+i, g] += 1.0 / len(group)
            
            # Add small perturbation to break symmetry
            if merge_strategy > 0:
                np.random.seed(42 + r + merge_strategy)
                U_init += np.random.randn(n*m, r) * 0.1
                V_init += np.random.randn(m*p, r) * 0.1
                W_init += np.random.randn(p*n, r) * 0.1
            
            result = optimize_from_init(U_init, V_init, W_init, r, max_iter=300)
            if result is not None:
                best_rank = r
                best_decomp = result
                found = True
                break
        
        if found:
            break
        
        # Strategy 2: Random initializations with various scales
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Sparse random: each column has few non-zeros
                U_init = np.zeros((n*m, r))
                V_init = np.zeros((m*p, r))
                W_init = np.zeros((p*n, r))
                for c in range(r):
                    # Pick 2-3 entries per column
                    num_nonzero_U = np.random.randint(2, 4)
                    num_nonzero_V = np.random.randint(2, 4)
                    num_nonzero_W = np.random.randint(2, 4)
                    
                    U_rows = np.random.choice(n*m, size=num_nonzero_U, replace=False)
                    V_rows = np.random.choice(m*p, size=num_nonzero_V, replace=False)
                    W_rows = np.random.choice(p*n, size=num_nonzero_W, replace=False)
                    
                    U_init[U_rows, c] = np.random.choice([-1, 1], size=num_nonzero_U)
                    V_init[V_rows, c] = np.random.choice([-1, 1], size=num_nonzero_V)
                    W_init[W_rows, c] = np.random.choice([-1, 1], size=num_nonzero_W)
            elif restart == 1:
                # Random with structure: initialize based on partial tensor structure
                U_init = np.zeros((n*m, r))
                V_init = np.zeros((m*p, r))
                W_init = np.zeros((p*n, r))
                np.random.seed(100 + r + restart)
                for c in range(r):
                    # Pick random i, j, k for this column
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U_init[i*m+j, c] = 1.0
                    V_init[j*p+k, c] = 1.0
                    W_init[k*n+i, c] = 1.0
                    # Add some additional entries
                    if np.random.random() < 0.5:
                        i2 = np.random.randint(0, n)
                        j2 = np.random.randint(0, m)
                        k2 = np.random.randint(0, p)
                        U_init[i2*m+j2, c] = 0.5
                        V_init[j2*p+k2, c] = 0.5
                        W_init[k2*n+i2, c] = 0.5
            elif restart == 2:
                # Dense random with small values
                scale = 0.3 + 0.1 * (restart % 5)
                np.random.seed(200 + r + restart)
                U_init = np.random.randn(n*m, r) * scale
                V_init = np.random.randn(m*p, r) * scale
                W_init = np.random.randn(p*n, r) * scale
            else:
                # Varying dense random
                scale = 0.2 + 0.15 * np.random.random()
                np.random.seed(300 + r + restart)
                U_init = np.random.randn(n*m, r) * scale
                V_init = np.random.randn(m*p, r) * scale
                W_init = np.random.randn(p*n, r) * scale
            
            result = optimize_from_init(U_init, V_init, W_init, r, max_iter=200)
            if result is not None:
                best_rank = r
                best_decomp = result
                found = True
                break
        
        if found:
            break
    
    # Final polish: if we have a near-solution with rank < 30, try to improve precision
    if best_decomp[0].shape[1] < 30:
        U, V, W = best_decomp
        r = U.shape[1]
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        obj, grad = make_objective_and_grad(r)
        try:
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < 1e-10:
                U_opt = res.x[0:size_U].reshape(n*m, r)
                V_opt = res.x[size_U:size_U + size_V].reshape(m*p, r)
                W_opt = res.x[size_U + size_V:].reshape(p*n, r)
                result = snap_and_verify(U_opt, V_opt, W_opt, 1e-5)
                if result is not None:
                    best_decomp = result
        except Exception:
            pass
    
    U_list = best_decomp[0].tolist()
    V_list = best_decomp[1].tolist()
    W_list = best_decomp[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
