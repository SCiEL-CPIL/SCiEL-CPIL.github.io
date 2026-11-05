import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We can represent T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def compute_residual(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Residual = sum_r U_r @ V_r @ W_r^T - T ?
        # No, the tensor is sum_r U_r \otimes V_r \otimes W_r.
        # We need to check if sum_r U[:, r] * V[:, r] * W[:, r] == T.
        # This is element-wise product summed over r.
        # T is 3D.
        # U[:, r] is 1D, V[:, r] is 1D, W[:, r] is 1D.
        # We can do:
        # Residual = np.einsum('ir,jr,kr->ijk', U, V, W) - T
        # But T is flattened? No, T is (n*m, m*p, p*n).
        # U is (n*m, R).
        # So np.einsum('ir,jr,kr->ijk', U, V, W) works.
        return np.linalg.norm(np.einsum('ir,jr,kr->ijk', U, V, W) - T)

    def snap_to_half_int(mat):
        return np.round(mat * 2) / 2

    def verify(U, V, W):
        # Check if entries are half-integers
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        # Check exactness
        res = np.einsum('ir,jr,kr->ijk', U, V, W) - T
        return np.linalg.norm(res) < 1e-9

    # Try to find rank 44
    start_time = time.time()
    deadline = start_time + 100 # 100 seconds
    
    # We will try to remove one term and optimize
    # We'll try a few random terms to remove
    best_result = None
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        rank = rank_school - 1
        # Initialize with schoolbook, remove term 'attempt'
        # To make it more interesting, we can remove different terms
        term_to_remove = attempt % rank_school
        
        U = U_school[:, :].copy()
        V = V_school[:, :].copy()
        W = W_school[:, :].copy()
        
        # Zero out the removed term
        U[:, term_to_remove] = 0
        V[:, term_to_remove] = 0
        W[:, term_to_remove] = 0
        
        # Add small noise to zero entries to help optimization
        # We only add noise to entries that are currently 0
        mask_U = (U == 0)
        mask_V = (V == 0)
        mask_W = (W == 0)
        
        U[mask_U] += np.random.normal(0, 0.1, size=np.sum(mask_U))
        V[mask_V] += np.random.normal(0, 0.1, size=np.sum(mask_V))
        W[mask_W] += np.random.normal(0, 0.1, size=np.sum(mask_W))
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*rank], x[n*m*rank:n*m*rank + m*p*rank], x[n*m*rank + m*p*rank:]
            U_opt = U_flat.reshape(n*m, rank)
            V_opt = V_flat.reshape(m*p, rank)
            W_opt = W_flat.reshape(p*n, rank)
            return compute_residual(U_opt, V_opt, W_opt)
        
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B is faster for large problems
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
        
        if result.fun < 1e-6:
            U_opt, V_opt, W_opt = result.x[:n*m*rank].reshape(n*m, rank), result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank), result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            U_snapped = snap_to_half_int(U_opt)
            V_snapped = snap_to_half_int(V_opt)
            W_snapped = snap_to_half_int(W_opt)
            
            if verify(U_snapped, V_snapped, W_snapped):
                best_result = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                break

    if best_result:
        return best_result
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

