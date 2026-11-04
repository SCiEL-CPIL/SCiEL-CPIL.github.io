import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    def is_valid(U, V, W, rank, atol=1e-6):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    # Strategy: Try to find a rank 41 decomposition (23 + 18)
    # If it fails, fallback to schoolbook 45.
    
    target_rank = 41
    if target_rank >= n * m * p:
        target_rank = n * m * p - 1

    best_solution, best_rank = get_schoolbook()
    
    # Try to improve
    # We use a structured initialization for rank 41
    # The first 23 columns correspond to M_3,3,3
    # The last 18 columns correspond to M_3,3,2
    
    def solve_rank_41():
        if time.time() > deadline:
            return None
        
        U = np.zeros((n * m, 41))
        V = np.zeros((m * p, 41))
        W = np.zeros((p * n, 41))
        
        # Part 2: M_3,3,2 (Last 18 columns)
        # This is just schoolbook for the last 2 columns of B (indices 3, 4 in p-dim)
        col_offset = 23
        count = 0
        for i in range(n):
            for j in range(m):
                for k in [3, 4]: # Last 2 columns
                    U[i * m + j, col_offset + count] = 1
                    V[j * p + k, col_offset + count] = 1
                    W[k * n + i, col_offset + count] = 1
                    count += 1
        # Now columns 23 to 40 are set for the M_3,3,2 part.
        
        # Part 1: M_3,3,3 (First 23 columns)
        # We need to find 23 terms for the first 3 columns.
        # Initialize with random sparse half-integers
        np.random.seed(42)
        for r in range(23):
            # Sparse initialization
            nnz = np.random.randint(2, 5)
            idx_u = np.random.choice(n * m, nnz, replace=False)
            idx_v = np.random.choice(m * p, nnz, replace=False)
            idx_w = np.random.choice(p * n, nnz, replace=False)
            
            vals = np.array([-1.0, -0.5, 0.5, 1.0])
            U[idx_u, r] = np.random.choice(vals, nnz)
            V[idx_v, r] = np.random.choice(vals, nnz)
            W[idx_w, r] = np.random.choice(vals, nnz)
            
            # Only affect the first 3 columns of V and W (k in 0,1,2)
            # Zero out V and W entries for k > 2 in these columns
            mask_v = np.arange(m * p).reshape(m, p)
            mask_w = np.arange(p * n).reshape(p, n)
            
            # Flatten masks
            # V[j, k] is non-zero only if k < 3
            # W[k, i] is non-zero only if k < 3
            
            # Reset V and W for k >= 3
            for j in range(m):
                for k in range(3, p):
                    V[j * p + k, r] = 0
            for k in range(3, p):
                for i in range(n):
                    W[k * n + i, r] = 0
            
            # Also ensure U doesn't leak? U is fine.

        # Optimize the first 23 columns using gradient descent
        # Objective: Maximize dot product with target tensor for first 3 columns?
        # Actually, we want Rec == 1 for all entries.
        # Since the last 18 columns are fixed and correct for k>2,
        # the first 23 columns must sum to 1 for k<3 and 0 for k>=3?
        # No, the last 18 columns contribute 0 to k<3?
        # Let's check:
        # V[j, k] for k in {3,4} is non-zero in last 18 cols.
        # W[k, i] for k in {3,4} is non-zero in last 18 cols.
        # U[i, j] is non-zero in last 18 cols.
        # Rec[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # For k in {3,4}, the first 23 cols have V[j,k,r]=0 and W[k,i,r]=0.
        # So the first 23 cols contribute 0 to k in {3,4}.
        # The last 18 cols contribute 1 to k in {3,4} (since they are schoolbook).
        # So k in {3,4} is satisfied.
        # For k in {0,1,2}, the last 18 cols have V[j,k,r]=0 and W[k,i,r]=0.
        # So the last 18 cols contribute 0 to k in {0,1,2}.
        # Thus, the first 23 cols MUST satisfy the M_3,3,3 multiplication.
        
        # We perform gradient descent on the first 23 columns to minimize error in M_3,3,3.
        
        max_iters = 500
        lr = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute Rec for first 3 columns
            Rec = np.zeros((n, m, 3))
            for r in range(23):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                # Only take k < 3
                Rec += np.einsum('ij,jk,ki->ijk', u, v[:, :3], w[:3, :], optimize=True)
            
            err = np.max(np.abs(Rec - 1.0))
            
            if err < 1e-6:
                # Check if values are close to half-integers
                # Snap
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                if is_valid(U_snapped, V_snapped, W_snapped, 41):
                    return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                else:
                    # Try to polish
                    pass

            # Gradient step
            err_vals = np.abs(Rec - 1.0)
            max_idx = np.unravel_index(np.argmax(err_vals), (n, m, 3))
            i_max, j_max, k_max = max_idx
            sign_err = np.sign(Rec[i_max, j_max, k_max] - 1.0)
            
            for r in range(23):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                
                # Gradient for U[i,j,r]
                # dU = sign_err * V[j,k] * W[k,i]
                # Note: V and W are full, but only k<3 parts matter for Rec
                grad_u = sign_err * v[j_max, k_max] * w[k_max, i_max]
                grad_v = sign_err * u[i_max, j_max] * w[k_max, i_max]
                grad_w = sign_err * u[i_max, j_max] * v[j_max, k_max]
                
                U[i_max * m + j_max, r] -= lr * grad_u
                V[j_max * p + k_max, r] -= lr * grad_v
                W[k_max * n + i_max, r] -= lr * grad_w
                
                # Keep V and W zero for k>=3
                # This is enforced by initialization, but gradient might introduce noise?
                # No, we only update V[j,k] where k is in 0..2 (from max_idx).
                # And W[k,i] where k is in 0..2.
        
        return None

    sol = solve_rank_41()
    if sol:
        return sol
    
    # Fallback
    return best_solution
