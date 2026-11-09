import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T as a sparse representation of indices
    # T[alpha, beta, gamma] = 1
    nz_alpha = []
    nz_beta = []
    nz_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                nz_alpha.append(alpha)
                nz_beta.append(beta)
                nz_gamma.append(gamma)
                
    nz_alpha = np.array(nz_alpha)
    nz_beta = np.array(nz_beta)
    nz_gamma = np.array(nz_gamma)
    N_nz = len(nz_alpha)
    
    def check_validity(U, V, W, rank):
        """
        Check if the decomposition is exact.
        """
        # Check non-zero entries
        U_nz = U[nz_alpha, :]
        V_nz = V[nz_beta, :]
        W_nz = W[nz_gamma, :]
        
        # Compute sum_r U V W for non-zeros
        prod = U_nz * V_nz * W_nz
        sum_prod = np.sum(prod, axis=1)
        if not np.allclose(sum_prod, 1.0, atol=1e-6):
            return False
        
        # Check a sample of zero entries to ensure no leakage
        # We can compute the full tensor if rank is small enough
        # T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This is dim_u * dim_v * dim_w * rank operations.
        # 6 * 21 * 14 * rank = 1764 * rank.
        # For rank 40, this is ~70k ops. Very fast.
        
        T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Create sparse target in dense form
        T_target = np.zeros((dim_u, dim_v, dim_w))
        T_target[nz_alpha, nz_beta, nz_gamma] = 1.0
        
        if not np.allclose(T_full, T_target, atol=1e-6):
            return False
            
        return True

    def als_step(U, V, W, rank, iters=50, tol=1e-10):
        """
        Perform ALS iterations.
        """
        # Pre-compute structures for efficient solving
        # For U: U[a, r] is involved in T[a, b, g]
        # We need to solve for each row U[a, :]
        # The equation is: sum_r U[a, r] V[b, r] W[g, r] = T[a, b, g]
        
        # This is a linear system for U[a, :]
        # Let X = U[a, :].
        # sum_r X_r (V[b, r] W[g, r]) = T[a, b, g]
        # This is not a simple dot product because V and W are coupled.
        # We can rewrite:
        # T[a, b, g] = sum_r X_r (V[b, r] W[g, r])
        # This is a matrix equation: X @ M = T_row?
        # No, the index r is summed.
        # Let's form a matrix A_a of size (N_constraints, rank)
        # where A_a[i, r] = V[b_i, r] W[g_i, r] for all (b_i, g_i) such that T[a, b_i, g_i] is constrained.
        # Then we solve A_a X^T = T_vec.
        
        # Since the tensor is sparse, for each 'a', there are only 'p' non-zero entries (for fixed i, j, k varies).
        # Wait, for fixed alpha = i*m+j, the non-zero betas are j*p+k for k=0..p-1.
        # So for each alpha, there are p non-zero entries in the tensor.
        # This means the linear system for each row of U is underdetermined if rank > p.
        # However, we have constraints from ALL non-zero entries.
        # Actually, ALS usually solves for one factor at a time, fixing the others.
        # If we fix V and W, we solve for U.
        # The objective is min_U || T - sum_r U_r V_r W_r ||^2.
        # This is a linear least squares problem for each entry of U?
        # No, U is a matrix.
        # Let's vectorize.
        # The term is sum_r U_{a,r} (V_{b,r} W_{g,r}).
        # For a fixed 'a', the variables are U_{a, 0..R-1}.
        # The constraints are for all (b, g) where T[a, b, g] is non-zero.
        # There are p such constraints for each 'a'.
        # If R > p, the system is underdetermined. We can use least squares (min norm solution) or regularize.
        # But we also have constraints from zero entries?
        # Standard ALS ignores zero entries or treats them as 0.
        # If we include zero entries, the system becomes very large (1764 constraints per row).
        # Let's include ALL entries to ensure zero leakage.
        
        # For each 'a' in 0..dim_u-1:
        #   Construct matrix M_a of shape (dim_v * dim_w, rank)
        #   M_a[idx, r] = V[b, r] * W[g, r] where idx = b * dim_w + g
        #   Construct vector T_a of shape (dim_v * dim_w)
        #   T_a[idx] = T[a, b, g]
        #   Solve M_a X^T = T_a for X = U[a, :]
        
        # This is expensive if done naively for every iteration.
        # But dim_v * dim_w = 21 * 14 = 294.
        # Rank is ~40.
        # 6 rows for U.
        # 21 rows for V.
        # 14 rows for W.
        # Total 41 linear systems of size 294 x 40.
        # This is very fast.
        
        def solve_for_U(U, V, W, rank):
            new_U = np.zeros_like(U)
            for a in range(dim_u):
                # Build M_a
                # M_a[b*dim_w + g, r] = V[b, r] * W[g, r]
                # We can use einsum to build this efficiently?
                # V is (dim_v, rank), W is (dim_w, rank)
                # Outer product of V and W along rank?
                # V[:, :, None] * W[None, :, :] -> (dim_v, dim_w, rank)
                # Reshape to (dim_v * dim_w, rank)
                M_a = (V[:, :, None] * W[None, :, :]).reshape(dim_v * dim_w, rank)
                
                # T_a
                # T[a, :, :] flattened
                # We need the target values.
                # We can precompute the full target tensor T_target once.
                T_a = T_target[a, :, :].flatten()
                
                # Solve M_a X^T = T_a
                # X is (rank,). M_a is (294, rank).
                # Use least squares.
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(M_a, T_a, rcond=None)
                    new_U[a, :] = sol
                except:
                    new_U[a, :] = U[a, :]
            return new_U

        def solve_for_V(U, V, W, rank):
            new_V = np.zeros_like(V)
            for b in range(dim_v):
                # M_b[a*dim_w + g, r] = U[a, r] * W[g, r]
                M_b = (U[:, :, None] * W[None, :, :]).reshape(dim_u * dim_w, rank)
                T_b = T_target[:, b, :].flatten()
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(M_b, T_b, rcond=None)
                    new_V[b, :] = sol
                except:
                    new_V[b, :] = V[b, :]
            return new_V

        def solve_for_W(U, V, W, rank):
            new_W = np.zeros_like(W)
            for g in range(dim_w):
                # M_g[a*dim_v + b, r] = U[a, r] * V[b, r]
                M_g = (U[:, :, None] * V[None, :, :]).reshape(dim_u * dim_v, rank)
                T_g = T_target[:, :, g].flatten()
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(M_g, T_g, rcond=None)
                    new_W[g, :] = sol
                except:
                    new_W[g, :] = W[g, :]
            return new_W

        # Precompute T_target
        T_target = np.zeros((dim_u, dim_v, dim_w))
        T_target[nz_alpha, nz_beta, nz_gamma] = 1.0

        prev_error = float('inf')
        for it in range(iters):
            U_old = U
            V_old = V
            W_old = W
            
            U = solve_for_U(U, V, W, rank)
            V = solve_for_V(U, V, W, rank)
            W = solve_for_W(U, V, W, rank)
            
            # Compute error
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.sum((T_est - T_target)**2)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
            
            # Check validity periodically
            if it % 5 == 0 and check_validity(U, V, W, rank):
                return U, V, W, True
                
        return U, V, W, check_validity(U, V, W, rank)

    # Try ranks from low to high
    # Schoolbook is 42.
    # Start with 20.
    
    for rank in range(20, 43):
        found = False
        # More restarts for low ranks
        n_restarts = 20 if rank < 30 else 10
        
        for restart in range(n_restarts):
            np.random.seed(42 + restart + rank)
            U_init = np.random.randn(dim_u, rank) * 0.1
            V_init = np.random.randn(dim_v, rank) * 0.1
            W_init = np.random.randn(dim_w, rank) * 0.1
            
            U, V, W, valid = als_step(U_init, V_init, W_init, rank)
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook if optimization fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
