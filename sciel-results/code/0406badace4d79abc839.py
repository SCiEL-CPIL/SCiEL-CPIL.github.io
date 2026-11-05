import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

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
        return U, V, W

    def get_rank50():
        """Construct a valid rank-50 decomposition using Strassen 2x2 blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7 with 2x2 Strassen blocks
        # A cols pairs: (0,1), (2,3)
        # B row pairs: (0,1), (2,3)
        # B col pairs: (0,1), (2,3), (4,5); col 6 separate
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def reduce_rank(U, V, W, max_iter=200):
        """
        Iteratively try to reduce rank by finding columns whose removal
        can be compensated by adjusting remaining columns.
        
        Key insight: T = sum_r U_r ⊗ V_r ⊗ W_r
        If we remove column r, we need to find delta adjustments to other
        columns such that the tensor is preserved. This is equivalent to
        checking if the residual tensor (after removing column r) has
        tensor rank 1 (i.e., can be expressed as a single rank-1 term).
        
        More practically: try to find a column r and coefficients such that
        the column can be absorbed. We use the fact that for each (a,b) slice,
        the matrix T[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
        
        A simpler approach: check if removing column r leaves a tensor that
        is still expressible with the remaining columns by solving a least-squares
        problem in the W-space for each (a,b) pair.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            best_removal = -1
            best_residual = np.inf
            
            for r in range(rank):
                if time.time() > deadline - 15:
                    break
                
                # Try removing column r
                # The residual is T - sum_{s != r} U_s ⊗ V_s ⊗ W_s
                # = U_r ⊗ V_r ⊗ W_r (the column we're removing)
                # We need to check if this can be expressed using the other columns.
                
                # For each (a, b), we need:
                # U[a,r] * V[b,r] * W[:,r] = sum_{s != r} alpha_{a,b,s} * U[a,s] * V[b,s] * W[:,s]
                # This is a linear system in alpha for each (a,b), but the alphas
                # must be consistent across (a,b) for a valid decomposition.
                
                # Simpler check: is the column r "redundant" in the sense that
                # U[:,r] ⊗ V[:,r] is in the span of {U[:,s] ⊗ V[:,s] : s != r}?
                # If so, and W[:,r] is in the appropriate span, we can remove it.
                
                # Build the matrix M where M[a,b,r] = U[a,r] * V[b,r]
                # Check if the r-th "column" (in the Kronecker sense) is dependent.
                
                # Actually, let's use a different approach: 
                # Check if the tensor contribution of column r can be absorbed.
                # For a valid absorption, we need vectors alpha_a (for each a) and 
                # beta_b (for each b) such that the cross-term structure works.
                
                # Practical heuristic: try to solve for adjustments to W
                # Given U and V fixed (minus column r), can we find W' such that
                # sum_{s != r} U[:,s] ⊗ V[:,s] ⊗ W'[:,s] = T?
                
                # This is a linear system: for each (a,b),
                # sum_{s != r} U[a,s] * V[b,s] * W'[s] = T[a,b,:]
                
                # Stack all (a,b) equations:
                # A * W' = T_flat
                # where A is a (n*m * m*p) x (rank-1) matrix with A[(a,b), s] = U[a,s]*V[b,s]
                
                # Build A for all columns except r
                keep = [s for s in range(rank) if s != r]
                if len(keep) == 0:
                    continue
                
                A = np.zeros((n * m * m * p, len(keep)))
                for idx_a in range(n * m):
                    for idx_b in range(m * p):
                        for s_idx, s in enumerate(keep):
                            A[idx_a * m * p + idx_b, s_idx] = U[idx_a, s] * V[idx_b, s]
                
                T_flat = T.reshape(n * m * m * p, p * n)
                
                # Solve least squares
                try:
                    W_prime, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat, rcond=1e-10)
                    # Check residual
                    recon = (A @ W_prime).reshape(n * m, m * p, p * n)
                    residual = np.max(np.abs(recon - T))
                    
                    if residual < 1e-8 and residual < best_residual:
                        best_residual = residual
                        best_removal = r
                except:
                    continue
            
            if best_removal >= 0:
                # Remove column best_removal and update W
                keep = [s for s in range(rank) if s != best_removal]
                U = U[:, keep]
                V = V[:, keep]
                
                # Recompute W using least squares
                A = np.zeros((n * m * m * p, len(keep)))
                for idx_a in range(n * m):
                    for idx_b in range(m * p):
                        for s_idx, s in enumerate(keep):
                            A[idx_a * m * p + idx_b, s_idx] = U[idx_a, s] * V[idx_b, s]
                
                T_flat = T.reshape(n * m * m * p, p * n)
                W_prime, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=1e-10)
                W = W_prime
                
                rank = len(keep)
            else:
                break
        
        return U, V, W, rank

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank using linear algebra
    U_red, V_red, W_red, final_rank = reduce_rank(U50, V50, W50)

    # Verify the reduced decomposition
    if verify_decomp(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    else:
        # Fallback to the verified rank-50
        U, V, W = U50, V50, W50

    # Final verification
    if not verify_decomp(U, V, W):
        U, V, W = get_schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
