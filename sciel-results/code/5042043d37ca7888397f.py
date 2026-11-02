import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def verify_and_snap(U, V, W, r):
        """Verify the decomposition and snap to half-integers."""
        # Try snapping to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        if np.allclose(T_recon, T_target, atol=1e-9):
            return U_snap, V_snap, W_snap
        
        # Try snapping to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        T_recon_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        
        if np.allclose(T_recon_int, T_target, atol=1e-9):
            return U_int, V_int, W_int
        
        return None

    def als_step(U, V, W, T, max_iter=2000, tol=1e-12):
        """
        Alternating Least Squares for Tensor Decomposition.
        Updates U, V, W in place (or returns new ones).
        """
        n_rows_U, r = U.shape
        n_rows_V, _ = V.shape
        n_rows_W, _ = W.shape
        
        prev_err = float('inf')
        for it in range(max_iter):
            # Update U
            # U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
            # This is T contracted with V and W
            # Using einsum: 'abc,br,cr->ar'
            U_new = np.einsum('abc,br,cr->ar', T, V, W)
            
            # Update V
            # V[b, r] = sum_{a,c} T[a,b,c] * U[a,r] * W[c,r]
            V_new = np.einsum('abc,ar,cr->br', T, U_new, W)
            
            # Update W
            # W[c, r] = sum_{a,b} T[a,b,c] * U[a,r] * V[b,r]
            W_new = np.einsum('abc,ar,br->cr', T, U_new, V_new)
            
            U, V, W = U_new, V_new, W_new
            
            # Calculate error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T, 'fro')
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    def try_rank_als(r, seed):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Run ALS
        U, V, W, err = als_step(U, V, W, T_target, max_iter=1000, tol=1e-10)
        
        if err < 1e-6:
            snapped = verify_and_snap(U, V, W, r)
            if snapped:
                return snapped
                
        return None

    # Try ranks from low to high
    # We know rank is <= 50.
    # Likely candidates: 25, 30, 35, 40, 45, 50
    # Let's try a range.
    ranks_to_try = [20, 22, 25, 28, 30, 35, 40, 45, 50]
    
    found = False
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple seeds
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            seed = 42 + attempt * 123 + target_rank * 7
            result = try_rank_als(target_rank, seed)
            
            if result:
                U_sol, V_sol, W_sol = result
                best_rank = target_rank
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
                found = True
                break
        
        if found:
            # If we found a solution, we can stop or try to go lower?
            # The loop goes low to high, so if we find one, it's the lowest in this list.
            # But maybe a lower rank exists that we missed due to bad init?
            # We can break here to save time, or continue to verify lower ranks if we skipped them.
            # Since we iterate low to high, the first success is the best in this list.
            break

    # If no solution found better than schoolbook, return schoolbook
    # But schoolbook is always valid.
    
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
