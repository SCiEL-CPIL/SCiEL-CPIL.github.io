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
    time_limit = 120  # Safety margin (kill at 150s)
    
    def compute_tensor(U, V, W, r):
        """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        T = np.zeros((n, m, p))
        for c in range(r):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            prod_uv = u_c[:, :, None] * v_c[None, :, :]
            w_tensor = w_c.T[:, None, :]
            T += prod_uv * w_tensor
        return T

    def is_valid(U, V, W, r):
        T = compute_tensor(U, V, W, r)
        return np.allclose(T, 1.0, atol=1e-8)

    def to_half_int_list(matrix):
        """Convert a numpy matrix to a list of lists of integer or half-integer floats."""
        out = []
        for row in matrix:
            rrow = []
            for val in row:
                fv = Fraction(val).limit_denominator(1000000)
                num, den = fv.numerator, fv.denominator
                if den == 1:
                    rrow.append(float(num))
                elif den == 2:
                    rrow.append(float(Fraction(num, 2)))
                else:
                    rrow.append(round(val * 2) / 2.0)
            out.append(rrow)
        return out

    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2

    def try_reduce_by_elimination(U, V, W, r, c_remove):
        """
        Try to eliminate term c_remove by adjusting the remaining r-1 terms.
        Returns (new_U, new_V, new_W, r-1) or None.
        """
        keep_idx = [i for i in range(r) if i != c_remove]
        r_keep = r - 1
        
        U_keep = U[:, keep_idx].copy()
        V_keep = V[:, keep_idx].copy()
        W_keep = W[:, keep_idx].copy()
        
        # Compute defect: what's missing from term c_remove
        T_keep = compute_tensor(U_keep, V_keep, W_keep, r_keep)
        b = 1.0 - T_keep
        b_flat = b.flatten()
        
        N_u = n * m
        N_v = m * p
        N_w = p * n
        N_vars = (N_u + N_v + N_w) * r_keep
        
        if N_vars < n * m * p:
            # Underdetermined - try least squares
            pass
        
        # Build Jacobian more efficiently
        J = np.zeros((n * m * p, N_vars))
        
        # Precompute for speed
        for idx, c in enumerate(keep_idx):
            u_c = U_keep[:, idx].reshape(n, m)
            v_c = V_keep[:, idx].reshape(m, p)
            w_c = W_keep[:, idx].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        
                        # d T[i,j,k] / d U[i,j,c] = V[j,k,c] * W[k,i,c]
                        J[eq_idx, (i * m + j) * r_keep + idx] = v_c[j, k] * w_c[k, i]
                        
                        # d T[i,j,k] / d V[j,k,c] = U[i,j,c] * W[k,i,c]
                        J[eq_idx, N_u * r_keep + (j * p + k) * r_keep + idx] = u_c[i, j] * w_c[k, i]
                        
                        # d T[i,j,k] / d W[k,i,c] = U[i,j,c] * V[j,k,c]
                        J[eq_idx, (N_u + N_v) * r_keep + (k * n + i) * r_keep + idx] = u_c[i, j] * v_c[j, k]
        
        # Solve least squares
        try:
            delta, residuals, rank_J, sv = np.linalg.lstsq(J, b_flat, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Check residual
        residual = np.linalg.norm(J @ delta - b_flat)
        if residual > 1e-6:
            return None
        
        # Apply delta
        U_new = U_keep.copy()
        V_new = V_keep.copy()
        W_new = W_keep.copy()
        
        delta_U = delta[:N_u * r_keep].reshape(N_u, r_keep)
        delta_V = delta[N_u * r_keep:(N_u + N_v) * r_keep].reshape(N_v, r_keep)
        delta_W = delta[(N_u + N_v) * r_keep:].reshape(N_w, r_keep)
        
        U_new += delta_U
        V_new += delta_V
        W_new += delta_W
        
        # Snap to half-integers
        U_snap = snap_to_half_int(U_new)
        V_snap = snap_to_half_int(V_new)
        W_snap = snap_to_half_int(W_new)
        
        # Verify
        if is_valid(U_snap, V_snap, W_snap, r_keep):
            return U_snap, V_snap, W_snap, r_keep
        
        # Try null space perturbations
        U_svd, s, Vh = np.linalg.svd(J, full_matrices=False)
        
        for _ in range(50):
            if time.time() - start_time > time_limit:
                return None
            
            z = np.random.randn(N_vars)
            z_row = Vh.T @ (Vh @ z)
            z_null = z - z_row
            
            for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
                delta_trial = delta + scale * z_null
                
                U_try = U_keep.copy()
                V_try = V_keep.copy()
                W_try = W_keep.copy()
                
                dU = delta_trial[:N_u * r_keep].reshape(N_u, r_keep)
                dV = delta_trial[N_u * r_keep:(N_u + N_v) * r_keep].reshape(N_v, r_keep)
                dW = delta_trial[(N_u + N_v) * r_keep:].reshape(N_w, r_keep)
                
                U_try += dU
                V_try += dV
                W_try += dW
                
                U_s = snap_to_half_int(U_try)
                V_s = snap_to_half_int(V_try)
                W_s = snap_to_half_int(W_try)
                
                if is_valid(U_s, V_s, W_s, r_keep):
                    return U_s, V_s, W_s, r_keep
        
        return None

    def try_reduce_rank(U, V, W, r):
        """Try to reduce rank by 1 by eliminating one term."""
        # Try each term for elimination
        for c_remove in range(r):
            if time.time() - start_time > time_limit:
                return None
            
            result = try_reduce_by_elimination(U, V, W, r, c_remove)
            if result:
                return result
        
        return None

    # Main Loop
    while time.time() - start_time < time_limit:
        result = try_reduce_rank(best_U, best_V, best_W, best_rank)
        if result:
            best_U, best_V, best_W, best_rank = result
            continue
        else:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
