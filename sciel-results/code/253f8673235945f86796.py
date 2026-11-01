import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 for all i, j, k
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the target tensor as a 3D array
    T = np.ones((n, m, p))
    
    # Function to compute reconstruction error
    def compute_error(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a tensor contraction.
        # We can do this efficiently with einsum
        # recon[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.linalg.norm(recon - T)
    
    # Function to snap to half-integers and verify
    def try_snap_and_verify(U, V, W, R):
        # Snap to nearest half-integer
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Verify exactness
        U_reshaped = U_snapped.reshape(n, m, R)
        V_reshaped = V_snapped.reshape(m, p, R)
        W_reshaped = W_snapped.reshape(p, n, R)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if np.allclose(recon, T, atol=1e-9):
            # Convert to lists of lists as required
            U_list = U_snapped.tolist()
            V_list = V_snapped.tolist()
            W_list = W_snapped.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    # Known rank-28 solution (Strassen based)
    def get_strassen_solution():
        R = 28
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen block for 2x2x2
        # Local indices i, j, k in {0, 1}
        # r in {0..6}
        U_loc = np.zeros((2, 2, 7))
        V_loc = np.zeros((2, 2, 7))
        W_loc = np.zeros((2, 2, 7))
        
        # r=0 (M1)
        U_loc[0, 0, 0] = 1; U_loc[1, 1, 0] = 1
        V_loc[0, 0, 0] = 1; V_loc[1, 1, 0] = 1
        W_loc[0, 0, 0] = 1; W_loc[1, 0, 0] = 1; W_loc[0, 1, 0] = 1; W_loc[1, 1, 0] = -1
        
        # r=1 (M2)
        U_loc[1, 0, 1] = 1; U_loc[1, 1, 1] = 1
        V_loc[0, 0, 1] = 1
        W_loc[0, 1, 1] = 1; W_loc[1, 1, 1] = -1
        
        # r=2 (M3)
        U_loc[0, 0, 2] = 1
        V_loc[1, 0, 2] = 1; V_loc[1, 1, 2] = -1
        W_loc[1, 0, 2] = 1; W_loc[1, 1, 2] = 1
        
        # r=3 (M4)
        U_loc[1, 1, 3] = 1
        V_loc[0, 1, 3] = 1; V_loc[0, 0, 3] = -1
        W_loc[0, 0, 3] = 1; W_loc[0, 1, 3] = 1
        
        # r=4 (M5)
        U_loc[0, 0, 4] = 1; U_loc[0, 1, 4] = 1
        V_loc[1, 1, 4] = 1
        W_loc[0, 0, 4] = -1; W_loc[1, 0, 4] = 1
        
        # r=5 (M6)
        U_loc[1, 0, 5] = 1; U_loc[0, 0, 5] = -1
        V_loc[0, 0, 5] = 1; V_loc[1, 0, 5] = 1
        W_loc[1, 1, 5] = 1
        
        # r=6 (M7)
        U_loc[0, 1, 6] = 1; U_loc[1, 1, 6] = -1
        V_loc[0, 1, 6] = 1; V_loc[1, 1, 6] = 1
        W_loc[0, 0, 6] = 1
        
        # Assemble 4 blocks
        blocks = [
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 2),
            (2, 2, 2),
        ]
        
        for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
            r_offset = b_idx * 7
            for i in range(2):
                for j in range(2):
                    U[i * m + (a_off + j), r_offset + 0: r_offset + 7] = U_loc[i, j, :]
            for j in range(2):
                for k in range(2):
                    V[(b_off + j) * p + (c_off + k), r_offset + 0: r_offset + 7] = V_loc[j, k, :]
            for k in range(2):
                for i in range(2):
                    W[(c_off + k) * n + i, r_offset + 0: r_offset + 7] = W_loc[k, i, :]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = get_strassen_solution()
    best_rank = 28

    # Try to find lower rank solutions using numerical optimization
    # We will try ranks 24, 25, 26, 27
    # Use a simple gradient-free or gradient-based search?
    # Given the non-convexity, random restarts with a local optimizer like L-BFGS-B or just simple gradient descent.
    # However, implementing a robust optimizer from scratch in 120s is risky.
    # Let's use a simple iterative refinement:
    # Fix two matrices, solve for the third via least squares?
    # This is the "Alternating Least Squares" (ALS) approach for tensor decomposition.
    
    def als_optimize(R, max_iters=50, n_restarts=5):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale to reasonable magnitude
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for W given U, V
                # Minimize || T - sum_r U[i,j,r] V[j,k,r] W[k,i,r] ||
                # This is a linear least squares problem for W.
                # Let's flatten the tensor equation.
                # For each (i, j, k), we have a target T[i,j,k] = 1.
                # The equation is: sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r] = 1
                # Let A_W be a matrix where each row corresponds to a (i,j,k) tuple.
                # Column r of A_W is (U[i,j,r] * V[j,k,r]).
                # We want A_W * W_flat = T_flat.
                # W_flat has size (p*n, R).
                
                # Construct A_W
                # Number of equations: n * m * p
                n_eq = n * m * p
                A_W = np.zeros((n_eq, R))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # W index is k*n + i
                            # The coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                            A_W[idx, :] = U[i * m + j, :] * V[j * p + k, :]
                            idx += 1
                
                T_flat = np.ones(n_eq)
                
                # Solve A_W * W_flat = T_flat
                try:
                    W_flat, residuals, rank, sv = np.linalg.lstsq(A_W, T_flat, rcond=None)
                    W = W_flat.reshape(p * n, R)
                except:
                    pass
                
                # Solve for V given U, W
                A_V = np.zeros((n_eq, R))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # V index is j*p + k
                            # Coefficient for V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r]
                            A_V[idx, :] = U[i * m + j, :] * W[k * n + i, :]
                            idx += 1
                try:
                    V_flat, residuals, rank, sv = np.linalg.lstsq(A_V, T_flat, rcond=None)
                    V = V_flat.reshape(m * p, R)
                except:
                    pass
                
                # Solve for U given V, W
                A_U = np.zeros((n_eq, R))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # U index is i*m + j
                            # Coefficient for U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r]
                            A_U[idx, :] = V[j * p + k, :] * W[k * n + i, :]
                            idx += 1
                try:
                    U_flat, residuals, rank, sv = np.linalg.lstsq(A_U, T_flat, rcond=None)
                    U = U_flat.reshape(n * m, R)
                except:
                    pass
                
                err = compute_error(U, V, W, R)
                if err < prev_err * 0.999:
                    prev_err = err
                else:
                    # Stagnation, break early
                    break
                
                if prev_err < 1e-10:
                    break
            
            if prev_err < 1e-8:
                # Try to snap
                sol = try_snap_and_verify(U, V, W, R)
                if sol:
                    return sol
        return None

    # Try ranks from 24 to 27
    for R in range(24, 28):
        if time.time() > deadline - 10:
            break
        sol = als_optimize(R, max_iters=30, n_restarts=10)
        if sol:
            best_solution = sol
            best_rank = R
            break # Found a lower rank, stop searching lower ones? 
            # Actually, we want the MINIMAL rank. So if we find R=24, we stop.
            # If we don't find R=24, we try R=25, etc.
            # The loop goes 24, 25, 26, 27.
            # If R=24 succeeds, we break.
            # If R=24 fails, we try R=25.
            # So we should iterate from low to high.

    # If no better solution found, return the rank-28 one
    return best_solution

