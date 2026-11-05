import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def make_tensor(n, m, p):
        """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = make_tensor(n, m, p)
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T. Returns True if valid."""
        rank = U.shape[1]
        # Compute sum over r of U[:,r] * V[:,r] * W[:,r]
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    s = 0.0
                    for r in range(rank):
                        s += U[i, r] * V[j, r] * W[k, r]
                    if abs(s - T[i, j, k]) > 1e-6:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using tensor contraction."""
        rank = U.shape[1]
        # Compute the tensor decomposition
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # We can compute this as: for each r, outer product U[:,r] * V[:,r] * W[:,r]
        computed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(computed, T, atol=1e-6)
    
    def verify_fastest(U, V, W):
        """Even faster verification using matrix multiplication."""
        rank = U.shape[1]
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Reshape and use matmul
        # For each pair (i,j), T[i,j,:] = (U[i,:] * V[j,:]) @ W.T
        # But this is still O(n*m*p*rank)
        
        # Alternative: compute T as einsum
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-6)
    
    def schoolbook(n, m, p):
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def try_iht(U, V, W, max_iters=500, tol=1e-8):
        """
        Iterative Hard Thresholding for sparse tensor decomposition.
        We try to reduce the effective rank by finding which rank-1 terms
        can be zeroed out while maintaining exact decomposition.
        
        Strategy: 
        1. Start with schoolbook (rank 72)
        2. For each term, try to zero it out and see if we can compensate
        3. Use alternating least squares to re-optimize non-zero terms
        """
        rank = U.shape[1]
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        # Track which terms are active
        active = np.ones(rank, dtype=bool)
        
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        best_rank = rank
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Try to remove each inactive-able term
            for r in range(rank):
                if not active[r]:
                    continue
                    
                # Temporarily deactivate term r
                active[r] = False
                
                # Try to re-solve with remaining terms using ALS
                success = _als_solve(T, n, m, p, U_curr, V_curr, W_curr, active, max_als_iters=20)
                
                if success:
                    # Check if the solution is valid
                    if verify_fastest(U_curr, V_curr, W_curr):
                        # Check if we can round to integers/half-integers
                        if _can_round(U_curr, V_curr, W_curr, active):
                            # Update best
                            best_rank = sum(active)
                            best_U = U_curr.copy()
                            best_V = V_curr.copy()
                            best_W = W_curr.copy()
                            # Keep term r deactivated
                        else:
                            # Restore
                            active[r] = True
                            _als_solve(T, n, m, p, U_curr, V_curr, W_curr, active, max_als_iters=10)
                    else:
                        active[r] = True
                        _als_solve(T, n, m, p, U_curr, V_curr, W_curr, active, max_als_iters=10)
                else:
                    active[r] = True
        
        return best_U, best_V, best_W, best_rank
    
    def _als_solve(T, n, m, p, U, V, W, active, max_als_iters=20):
        """
        Alternating Least Squares to optimize U, V, W for active terms only.
        Returns True if convergence achieved.
        """
        rank = len(active)
        active_idx = np.where(active)[0]
        
        for it in range(max_als_iters):
            if time.time() > deadline - 5.0:
                return False
            
            # Update U: for each i, minimize ||sum_r U[i,r] * V[:,r] * W[:,r] - T[i,:,:]||
            # This is: sum_r U[i,r] * outer(V[:,r], W[:,r]) = T[i,:,:].flatten()
            # So: V_r @ W_r^T * U[i,r] = T_i
            # Let A_i = [outer(V[:,r1], W[:,r1]), ..., outer(V[:,rn], W[:,rn])]
            # Then A_i @ U[i, active] = T[i,:,:].flatten()
            
            for i in range(n*m):
                # Build the design matrix
                cols = []
                for r in active_idx:
                    cols.append(np.outer(V[:, r], W[:, r]))
                if len(cols) == 0:
                    continue
                A = np.column_stack(cols)  # shape (m*p*p*n, num_active)
                b = T[i, :, :].flatten()
                # Solve least squares
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i, active_idx] = sol
                except:
                    pass
            
            # Update V: for each j, minimize ||sum_r V[j,r] * U[:,r] * W[:,r] - T[:,j,:]||
            for j in range(m*p):
                cols = []
                for r in active_idx:
                    cols.append(np.outer(U[:, r], W[:, r]))
                if len(cols) == 0:
                    continue
                A = np.column_stack(cols)  # shape (n*m*p*n, num_active)
                b = T[:, j, :].flatten()
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j, active_idx] = sol
                except:
                    pass
            
            # Update W: for each k, minimize ||sum_r W[k,r] * U[:,r] * V[:,r] - T[:,:,k]||
            for k in range(p*n):
                cols = []
                for r in active_idx:
                    cols.append(np.outer(U[:, r], V[:, r]))
                if len(cols) == 0:
                    continue
                A = np.column_stack(cols)  # shape (n*m*m*p, num_active)
                b = T[:, :, k].flatten()
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k, active_idx] = sol
                except:
                    pass
            
            # Check convergence
            if it > 0:
                computed = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(computed - T))
                if err < 1e-8:
                    return True
        
        # Final check
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.max(np.abs(computed - T))
        return err < 1e-6
    
    def _can_round(U, V, W, active, tol=1e-6):
        """Check if the active entries can be rounded to integers or half-integers."""
        for r in np.where(active)[0]:
            for i in range(n*m):
                if abs(U[i, r]) > tol:
                    # Check if close to integer or half-integer
                    val = U[i, r]
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) > tol and abs(val - nearest_half) > tol:
                        return False
            for j in range(m*p):
                if abs(V[j, r]) > tol:
                    val = V[j, r]
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) > tol and abs(val - nearest_half) > tol:
                        return False
            for k in range(p*n):
                if abs(W[k, r]) > tol:
                    val = W[k, r]
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) > tol and abs(val - nearest_half) > tol:
                        return False
        return True
    
    def round_entries(U, V, W, active, tol=1e-6):
        """Round entries to nearest integer or half-integer."""
        U_r = U.copy()
        V_r = V.copy()
        W_r = W.copy()
        
        for r in np.where(active)[0]:
            for i in range(n*m):
                if abs(U_r[i, r]) > tol:
                    val = U_r[i, r]
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) <= abs(val - nearest_half):
                        U_r[i, r] = float(nearest_int)
                    else:
                        U_r[i, r] = nearest_half
            for j in range(m*p):
                if abs(V_r[j, r]) > tol:
                    val = V_r[j, r]
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) <= abs(val - nearest_half):
                        V_r[j, r] = float(nearest_int)
                    else:
                        V_r[j, r] = nearest_half
            for k in range(p*n):
                if abs(W_r[k, r]) > tol:
                    val = W_r[k, r]
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_int) <= abs(val - nearest_half):
                        W_r[k, r] = float(nearest_int)
                    else:
                        W_r[k, r] = nearest_half
        return U_r, V_r, W_r
    
    # Start with schoolbook
    U, V, W = schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try IHT to reduce rank
    U_iht, V_iht, W_iht, rank_iht = try_iht(U, V, W, max_iters=100)
    
    if rank_iht < best_rank:
        # Verify the IHT solution
        if verify_fastest(U_iht, V_iht, W_iht):
            # Round to integers/half-integers
            active = np.ones(U_iht.shape[1], dtype=bool)
            # Determine which terms are actually used
            for r in range(U_iht.shape[1]):
                if np.max(np.abs(U_iht[:, r])) < 1e-6 and \
                   np.max(np.abs(V_iht[:, r])) < 1e-6 and \
                   np.max(np.abs(W_iht[:, r])) < 1e-6:
                    active[r] = False
            
            U_rounded, V_rounded, W_rounded = round_entries(U_iht, V_iht, W_iht, active)
            
            # Verify rounded solution
            if verify_fastest(U_rounded, V_rounded, W_rounded):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = sum(active)
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    # Remove zero columns to minimize rank
    def remove_zero_cols(U_l, V_l, W_l):
        rank = len(U_l[0]) if U_l else 0
        keep = []
        for r in range(rank):
            u_col = [row[r] for row in U_l]
            v_col = [row[r] for row in V_l]
            w_col = [row[r] for row in W_l]
            if any(abs(x) > 1e-9 for x in u_col) or \
               any(abs(x) > 1e-9 for x in v_col) or \
               any(abs(x) > 1e-9 for x in w_col):
                keep.append(r)
        
        U_new = [[row[r] for r in keep] for row in U_l]
        V_new = [[row[r] for r in keep] for row in V_l]
        W_new = [[row[r] for r in keep] for row in W_l]
        return U_new, V_new, W_new
    
    U_final, V_final, W_final = remove_zero_cols(U_list, V_list, W_list)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
