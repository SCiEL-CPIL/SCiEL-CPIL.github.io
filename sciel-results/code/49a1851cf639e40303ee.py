import numpy as np
import time
import itertools
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 145.0 # 5s margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((N, rank_sb))
    V_sb = np.zeros((M, rank_sb))
    W_sb = np.zeros((P, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Target tensor T of shape (N, M, P) flattened?
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k'
    # Actually, the tensor is sparse.
    # We want sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] == T[i*m+j, j'*p+k, k'*n+i']
    
    # Construct the sparse indices for the target tensor
    # The non-zero entries are at (i*m+j, j*p+k, k*n+i) for i in n, j in m, k in p
    # Value is 1.
    
    target_indices = []
    target_values = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_indices.append((idx_u, idx_v, idx_w))
                target_values.append(1.0)
                
    def reconstruct(U, V, W):
        # Compute the tensor from U, V, W
        # result[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We only care about the specific indices in target_indices?
        # No, for exactness, the result must match 0 everywhere else.
        # But the tensor rank decomposition implies the whole tensor matches.
        # If we only check the 1s, we might miss constraints.
        # However, the tensor is defined by these 1s and 0s.
        # If we minimize error over ALL entries, it's a large matrix.
        # N=12, M=20, P=15. Total 3600 entries.
        # This is small enough to handle fully.
        
        # Tensor U is (N, R), V is (M, R), W is (P, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        # Sum over R:
        # T_ijk = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Use einsum
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T
        
    def check_error(U, V, W):
        T = reconstruct(U, V, W)
        # Target T is 1 at (i, j, k) and 0 elsewhere
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1.0
        
        err = np.sum((T - T_target)**2)
        return err, T
        
    def try_rank(R):
        # Random initialization
        best_err = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        # Number of restarts based on remaining time
        # Each optimization step might take ~0.1s to 1s
        # Let's try a few simple gradient steps or just random search with local polish
        # For small R, random search with local refinement can work.
        
        # Since we need EXACT integer/half-integer, we can use a grid search on a small lattice?
        # No, R is up to 49. Dimension of U is 12*49 = 588. Too large for grid.
        
        # Use scipy.optimize?
        # We can flatten U, V, W into a single vector x of length R*(N+M+P)
        
        from scipy.optimize import minimize
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
        def unflatten(x):
            size_U = N * R
            size_V = M * R
            U = x[:size_U].reshape(N, R)
            V = x[size_U:size_U+size_V].reshape(M, R)
            W = x[size_U+size_V:].reshape(P, R)
            return U, V, W
            
        def objective(x):
            U, V, W = unflatten(x)
            err, _ = check_error(U, V, W)
            return err
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > DEADLINE - 10.0:
                break
                
            # Random init
            U0 = np.random.randn(N, R) * 0.1
            V0 = np.random.randn(M, R) * 0.1
            W0 = np.random.randn(P, R) * 0.1
            
            x0 = flatten(U0, V0, W0)
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-5})
            
            if res.fun < best_err:
                best_err = res.fun
                best_U, best_V, best_W = unflatten(res.x)
                
        # If we found a very small error, try to snap to rationals
        if best_err < 1e-8:
            U_int, V_int, W_int = best_U, best_V, best_W
            # Try to snap to half-integers (multiply by 2, round, divide by 2)
            # Or integers
            # Check if snapping works
            U_snap = np.round(U_int * 2) / 2.0
            V_snap = np.round(V_int * 2) / 2.0
            W_snap = np.round(W_int * 2) / 2.0
            err_snap, _ = check_error(U_snap, V_snap, W_snap)
            if err_snap < 1e-8:
                return U_snap, V_snap, W_snap, R
            else:
                # Try integer snap
                U_snap = np.round(U_int)
                V_snap = np.round(V_int)
                W_snap = np.round(W_int)
                err_snap, _ = check_error(U_snap, V_snap, W_snap)
                if err_snap < 1e-8:
                    return U_snap, V_snap, W_snap, R
        return None

    # Try ranks from 49 down to 40
    # 49 is a known upper bound for 4x4x4, so it's a strong candidate
    # For 3x4x5, it might be lower, but 49 is safe.
    for R in range(49, 39, -1):
        if time.time() > DEADLINE - 20.0:
            break
        sol = try_rank(R)
        if sol:
            U, V, W, r = sol
            # Convert to list of lists as required
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
