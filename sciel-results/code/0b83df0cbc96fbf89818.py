import numpy as np
from scipy.optimize import least_squares
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Build target tensor
    # T[a, b, c] = 1 if indices are consistent
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    targets_a = []
    targets_b = []
    targets_c = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_a.append(a)
                targets_b.append(b)
                targets_c.append(c)
                
    targets_a = np.array(targets_a, dtype=int)
    targets_b = np.array(targets_b, dtype=int)
    targets_c = np.array(targets_c, dtype=int)
    
    def verify_solution(U, V, W):
        """Verify if the solution is exact"""
        # U: (6, R), V: (15, R), W: (10, R)
        # Compute T_full
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check target entries
        T_target_vals = T_full[targets_a, targets_b, targets_c]
        if not np.allclose(T_target_vals, 1.0, atol=1e-9):
            return False
            
        # Check zero entries
        # Create a mask of all indices
        all_a = np.arange(size_U)
        all_b = np.arange(size_V)
        all_c = np.arange(size_W)
        
        # We need to check entries NOT in targets
        # It's faster to check the whole tensor against a target tensor
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_a, targets_b, targets_c] = 1.0
        
        max_err = np.max(np.abs(T_full - T_target))
        return max_err < 1e-9

    def get_schoolbook():
        """Return schoolbook decomposition (Rank 30)"""
        R = n * m * p
        U = np.zeros((size_U, R))
        V = np.zeros((size_V, R))
        W = np.zeros((size_W, R))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def try_reduce_rank(R, num_attempts=50):
        """
        Try to find a decomposition of rank R by perturbing the schoolbook solution.
        We keep R columns from the schoolbook and optimize them.
        """
        U_sb, V_sb, W_sb = get_schoolbook()
        R_sb = U_sb.shape[1] # 30
        
        # Precompute target tensor for faster residual calculation
        # We can just use the sparse check in verify, but for optimization we need residuals
        # Residuals: diff between T_full and T_target
        
        # To speed up optimization, we can flatten the target
        T_target = np.zeros((size_U, size_V, size_W))
        T_target[targets_a, targets_b, targets_c] = 1.0
        T_target_flat = T_target.flatten()
        
        def residuals_from_flat(x):
            U = x[:size_U*R].reshape(size_U, R)
            V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
            W = x[(size_U+size_V)*R:].reshape(size_W, R)
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_full - T_target).flatten()

        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
                
            # Select R columns to keep from the schoolbook
            # This creates a "warm start" where the initial guess is very close to a valid solution
            # for the subset of terms, but we allow the values to change.
            kept_indices = np.random.choice(R_sb, R, replace=False)
            
            # Initialize with schoolbook values for kept indices, 0 otherwise
            U_init = np.zeros((size_U, R))
            V_init = np.zeros((size_V, R))
            W_init = np.zeros((size_W, R))
            
            for new_idx, old_idx in enumerate(kept_indices):
                U_init[:, new_idx] = U_sb[:, old_idx]
                V_init[:, new_idx] = V_sb[:, old_idx]
                W_init[:, new_idx] = W_sb[:, old_idx]
                
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    residuals_from_flat,
                    x0,
                    method='trf',
                    max_nfev=100, # Keep it low to allow many restarts
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10
                )
                
                if res.cost < 1e-8:
                    x_sol = res.x
                    # Snap to half-integers
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    if verify_solution(U, V, W):
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
                
        return None

    start_time = time.time()
    deadline = 110 # Safety margin
    
    # Try to find a rank lower than 30
    # Known improvements might be 29 or 28
    for R in [28, 29]:
        if time.time() > deadline:
            break
        result = try_reduce_rank(R, num_attempts=100)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
