"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for return
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1.0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def snap_to_half_integers(mat):
        """Snap values to nearest 0.5 increment."""
        return np.round(mat * 2) / 2.0

    def verify_snapped(U, V, W):
        """Verify the snapped decomposition is exact."""
        if not check_decomposition(U, V, W, tol=1e-9):
            return False
        # Check if all values are effectively half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        return True

    # Strategy 1: Construct a known low-rank decomposition (Rank 21)
    # Based on the fact that M_{2,3,1} has rank 3.
    # For M_{2,3,7}, we can use 3 ranks per column of B, total 21.
    rank_21 = 21
    U_21 = np.zeros((nm, rank_21))
    V_21 = np.zeros((mp, rank_21))
    W_21 = np.zeros((pn, rank_21))
    
    for k in range(p):
        for j in range(m):
            r = k * m + j
            # U index: i*m + j. Non-zero for all i.
            # U[i*m + j, r] = 1
            for i in range(n):
                U_21[i*m + j, r] = 1.0
            # V index: j*p + k. Non-zero for this j, k.
            # V[j*p + k, r] = 1
            V_21[j*p + k, r] = 1.0
            # W index: k*n + i. Non-zero for all i.
            # W[k*n + i, r] = 1
            for i in range(n):
                W_21[k*n + i, r] = 1.0

    if verify_snapped(U_21, V_21, W_21):
        best_rank = rank_21
        best_U = U_21
        best_V = V_21
        best_W = W_21
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1.0
                    V_school[j*p + k, col] = 1.0
                    W_school[k*n + i, col] = 1.0
                    col += 1
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school

    # Try to find even lower rank using search
    # We can try ranks below best_rank
    # Given the time, we might not find one, but we can try a few.
    # The search is expensive, so we limit it.
    
    def als_iteration(U, V, W, T, max_iter=20, tol=1e-12):
        nm, mp, pn = T.shape
        r = U.shape[1]
        
        def compute_residual(U, V, W):
            return np.einsum('ac,bc,dc->abd', U, V, W) - T

        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            X_U = np.zeros((mp * pn, r))
            for r_idx in range(r):
                outer = np.outer(V[:, r_idx], W[:, r_idx])
                X_U[:, r_idx] = outer.flatten()
                
            for i in range(nm):
                y = T[i, :, :].flatten()
                sol, residuals, rank, sv = np.linalg.lstsq(X_U, y, rcond=None)
                U[i, :] = sol
            
            # Update V
            X_V = np.zeros((nm * pn, r))
            for r_idx in range(r):
                outer = np.outer(U[:, r_idx], W[:, r_idx])
                X_V[:, r_idx] = outer.flatten()
                
            for j in range(mp):
                y = T[:, j, :].flatten()
                sol, residuals, rank, sv = np.linalg.lstsq(X_V, y, rcond=None)
                V[j, :] = sol

            # Update W
            X_W = np.zeros((nm * mp, r))
            for r_idx in range(r):
                outer = np.outer(U[:, r_idx], V[:, r_idx])
                X_W[:, r_idx] = outer.flatten()
                
            for k in range(pn):
                y = T[:, :, k].flatten()
                sol, residuals, rank, sv = np.linalg.lstsq(X_W, y, rcond=None)
                W[k, :] = sol

            # Check convergence
            err = np.linalg.norm(compute_residual(U, V, W))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    # Search for rank < best_rank
    # Try a few random initializations for ranks 20, 19, ... down to 15
    # This is a heuristic and might not succeed, but it's better than nothing.
    for target_rank in range(best_rank - 1, 14, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Limit attempts to save time
        num_attempts = 5
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, T, max_iter=30)
            
            # Check if it's a good solution
            if check_decomposition(U, V, W, tol=1e-8):
                # Try to snap to half-integers
                U_snapped = snap_to_half_integers(U)
                V_snapped = snap_to_half_integers(V)
                W_snapped = snap_to_half_integers(W)
                
                if verify_snapped(U_snapped, V_snapped, W_snapped):
                    best_rank = target_rank
                    best_U = U_snapped
                    best_V = V_snapped
                    best_W = W_snapped
                    found = True
                    break
                else:
                    pass
        
        if found:
            break

    # Convert to lists
    def to_list(mat):
        return [[0 if abs(x) < 1e-12 else (1 if abs(x-1) < 1e-12 else (-1 if abs(x+1) < 1e-12 else (0.5 if abs(x-0.5) < 1e-12 else (-0.5 if abs(x+0.5) < 1e-12 else x)))) for x in row] for row in mat]
        
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
