import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank
    
    start_time = time.time()
    time_limit = 140 # Safety margin
    
    def is_valid(U, V, W, r):
        # Check if sum U*V*W == 1 for all i,j,k
        # Vectorized check
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We need T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient way:
        # For each c, we have a rank 1 contribution.
        # We can sum them up.
        # But looping over r is fast if r is small.
        # r is ~100. n*m*p is 105.
        # 105 * 100 = 10500 ops. Very fast.
        
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            # Contribution is u_c[i,j] * v_c[j,k] * w_c[k,i]
            # This is an outer product-like structure but cyclic.
            # We can compute:
            # Temp[i, j, k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
            # We can do:
            # A[i, j, k] = u_c[i,j] * v_c[j,k]
            # T[i, j, k] += A[i, j, k] * w_c[k, i]
            
            # Vectorized:
            # u_c is (n, m), v_c is (m, p).
            # prod_uv[i, j, k] = u_c[i, j] * v_c[j, k]
            # This can be done by broadcasting: u_c[:, :, None] * v_c[None, :, :]
            # Shape (n, m, p)
            prod_uv = u_c[:, :, None] * v_c[None, :, :]
            # Now multiply by w_c[k, i]
            # w_c is (p, n). We need to align k and i.
            # prod_uv[i, j, k] * w_c[k, i]
            # We can transpose w_c to (n, p) and then take diagonal?
            # Or simply:
            # w_c_T = w_c.T # (n, p)
            # We need w_c[k, i] for each i, k.
            # This is the element-wise product of prod_uv and a tensor formed by w_c.
            # Let's construct the tensor for w_c:
            # w_tensor[i, j, k] = w_c[k, i]
            # This is independent of j.
            # So w_tensor = w_c.T[None, :, :] broadcast to (n, m, p)?
            # w_c.T is (n, p).
            # We want w_c[k, i].
            # Let's swap axes of w_c.T?
            # w_c is (p, n).
            # We need index [k, i].
            # So we want a tensor W_t where W_t[i, j, k] = w_c[k, i].
            # W_t = w_c.T[:, None, :] # (n, 1, p) -> w_c.T[i, k] is w_c[k, i].
            # Wait, w_c.T[i, k] = w_c[k, i]. Yes.
            # So W_t = w_c.T[:, None, :]
            # prod_uv * W_t will broadcast correctly.
            
            w_tensor = w_c.T[:, None, :]
            T += prod_uv * w_tensor
            
        return np.allclose(T, 1.0, atol=1e-6)

    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2

    def try_reduce_rank(U, V, W, r):
        """
        Attempts to reduce rank by 1 by removing one term and solving for the rest.
        Returns (new_U, new_V, new_W, new_rank) or None.
        """
        # Try removing each term
        for c_remove in range(r):
            if time.time() - start_time > time_limit:
                return None
            
            # Create indices of terms to keep
            keep_idx = [i for i in range(r) if i != c_remove]
            
            # Extract current matrices for kept terms
            U_keep = U[:, keep_idx]
            V_keep = V[:, keep_idx]
            W_keep = W[:, keep_idx]
            
            # Compute current sum T_keep
            # We need the defect b = 1 - T_keep
            # T_keep is computed similarly to is_valid but only for kept terms
            T_keep = np.zeros((n, m, p))
            for idx, c in enumerate(keep_idx):
                u_c = U_keep[:, idx].reshape(n, m)
                v_c = V_keep[:, idx].reshape(m, p)
                w_c = W_keep[:, idx].reshape(p, n)
                prod_uv = u_c[:, :, None] * v_c[None, :, :]
                w_tensor = w_c.T[:, None, :]
                T_keep += prod_uv * w_tensor
            
            b = 1.0 - T_keep
            b_flat = b.flatten() # Size n*m*p = 105
            
            # Build Jacobian J
            # Variables: U_keep (n*m * (r-1)), V_keep (m*p * (r-1)), W_keep (p*n * (r-1))
            # Total vars N_v = (n*m + m*p + p*n) * (r-1)
            
            N_u = n * m
            N_v = m * p
            N_w = p * n
            N_vars = (N_u + N_v + N_w) * (r - 1)
            
            J = np.zeros((n * m * p, N_vars))
            
            # Populate J
            # Variable index mapping:
            # 0 .. N_u*(r-1)-1 : U
            # N_u*(r-1) .. N_u*(r-1) + N_v*(r-1)-1 : V
            # ... : W
            
            # For each equation (i, j, k):
            # d T[i,j,k] / d U[i,j,c] = V[j,k,c] * W[k,i,c]
            # d T[i,j,k] / d V[j,k,c] = U[i,j,c] * W[k,i,c]
            # d T[i,j,k] / d W[k,i,c] = U[i,j,c] * V[j,k,c]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        
                        for idx, c in enumerate(keep_idx):
                            # U derivative
                            val_U = V_keep[j * p + k, idx] * W_keep[k * n + i, idx]
                            var_idx_U = i * m + j
                            J[eq_idx, var_idx_U * (r - 1) + idx] = val_U
                            
                            # V derivative
                            val_V = U_keep[i * m + j, idx] * W_keep[k * n + i, idx]
                            var_idx_V = j * p + k
                            J[eq_idx, (N_u * (r - 1)) + var_idx_V * (r - 1) + idx] = val_V
                            
                            # W derivative
                            val_W = U_keep[i * m + j, idx] * V_keep[j * p + k, idx]
                            var_idx_W = k * n + i
                            J[eq_idx, (N_u + N_v) * (r - 1) + var_idx_W * (r - 1) + idx] = val_W
            
            # Solve J delta = b_flat
            # Use lstsq for min norm solution
            try:
                delta_p, residuals, rank_J, sv = np.linalg.lstsq(J, b_flat, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check if residual is small
            if np.linalg.norm(J.dot(delta_p) - b_flat) > 1e-5:
                continue
                
            # Get null space basis
            # sv contains singular values. rank_J is the effective rank.
            # The null space is spanned by the right singular vectors corresponding to zero singular values.
            # V from SVD is (N_vars, N_vars) if full, or (N_vars, min(M, N)) if reduced.
            # np.linalg.lstsq doesn't return V.
            # We need to compute SVD of J to get V.
            # J is 105 x N_vars.
            U_svd, s, Vh = np.linalg.svd(J, full_matrices=False)
            # Vh is (min(105, N_vars), N_vars) = (105, N_vars)
            # The rows of Vh are the basis for the row space.
            # The null space is the orthogonal complement.
            
            # We can generate random vectors in the null space by:
            # z ~ N(0, I)
            # z_null = z - Vh.T * (Vh * z)
            
            # We want to find a solution delta = delta_p + z_null * alpha
            # that snaps well.
            
            # Let's try a few random alpha
            found = False
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    return None
                    
                alpha = np.random.randn(105)
                # Project alpha onto row space? No, we need a vector in null space.
                # Let z be a random vector of size N_vars.
                z = np.random.randn(N_vars)
                # Project z onto row space: Vh.T * (Vh * z)
                # Vh is (105, N_vars)
                # Vh * z is (105,)
                # Vh.T * (Vh * z) is (N_vars,)
                z_row = Vh.T.dot(Vh.dot(z))
                z_null = z - z_row
                
                delta = delta_p + z_null
                # Scale z_null? The magnitude of z_null is arbitrary.
                # We might want to try different scales.
                # But delta_p is the min norm solution. z_null adds "gauge" freedom.
                # The snapping depends on the absolute values.
                # We can try scaling z_null.
                
                # Apply delta
                U_new = U_keep.copy()
                V_new = V_keep.copy()
                W_new = W_keep.copy()
                
                # Split delta
                delta_U = delta[:N_u * (r - 1)].reshape(N_u, r - 1)
                delta_V = delta[N_u * (r - 1): (N_u + N_v) * (r - 1)].reshape(N_v, r - 1)
                delta_W = delta[(N_u + N_v) * (r - 1):].reshape(N_w, r - 1)
                
                U_new += delta_U
                V_new += delta_V
                W_new += delta_W
                
                # Snap
                U_snapped = snap_to_half_int(U_new)
                V_snapped = snap_to_half_int(V_new)
                W_snapped = snap_to_half_int(W_new)
                
                # Check validity
                if is_valid(U_snapped, V_snapped, W_snapped, r - 1):
                    found = True
                    return U_snapped, V_snapped, W_snapped, r - 1
            
            if found:
                break
                
        return None

    # Main Loop
    while time.time() - start_time < time_limit:
        result = try_reduce_rank(best_U, best_V, best_W, best_rank)
        if result:
            best_U, best_V, best_W, best_rank = result
            # Try to reduce again
            continue
        else:
            # No reduction found
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
