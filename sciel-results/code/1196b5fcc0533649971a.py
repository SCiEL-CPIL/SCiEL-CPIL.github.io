import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    # Deadline guard
    start_time = time.time()
    deadline = 110  # Keep 10s margin for safety
    
    def make_schoolbook(rank):
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook(rank_school)
    best_rank = rank_school
    T = np.ones((n, m, p), dtype=np.float64)
    T_flat = T.flatten()

    def compute_tensor_sum(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # einsum: 'ijr,jkr,kir->ijk'
        return np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)

    def try_rank(R, max_sweeps=200, max_restarts=5):
        nonlocal best_U, best_V, best_W, best_rank
        nonlocal deadline, start_time
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return False
            
            # Initialize with schoolbook terms 0..R-1
            U, V, W = make_schoolbook(R)
            
            # Add small random perturbation to escape local minima
            if restart > 0:
                scale = 0.05 * restart
                U += np.random.normal(0, scale, U.shape)
                V += np.random.normal(0, scale, V.shape)
                W += np.random.normal(0, scale, W.shape)
            
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            tensor_sum = compute_tensor_sum(U, V, W)
            error = np.sum((tensor_sum - T) ** 2)
            
            for sweep in range(max_sweeps):
                if time.time() > deadline:
                    return False
                
                # Update U
                for r in range(R):
                    # Residual excluding r
                    contrib_r = U_r[:, :, r] * V_r[:, :, r] * W_r[:, :, r]
                    residual = T - tensor_sum + contrib_r
                    residual_flat = residual.flatten()
                    
                    # Construct M_U for this r
                    # M_U[i,j,k, i',j'] = delta_ii' delta_jj' V_r[j,k] W_r[k,i]
                    # We need to solve M_U u = residual_flat
                    # M_U is 120x20.
                    # Instead of building M_U explicitly, we can use the structure.
                    # The system is: sum_{j,k} V_r[j,k] W_r[k,i] U_r[i,j] = residual[i,j,k] ? No.
                    # The term is U_r[i,j] V_r[j,k] W_r[k,i].
                    # For fixed i,j, the equation is:
                    # U_r[i,j] * sum_k V_r[j,k] W_r[k,i] = residual[i,j,k] ?? No, residual depends on k.
                    # The least squares problem is:
                    # min_U sum_{i,j,k} (U[i,j] V[j,k] W[k,i] - R[i,j,k])^2
                    # This is not separable by (i,j) because R depends on k and V,W depend on k.
                    # However, for fixed i,j, let u = U[i,j].
                    # term = u * V[j,k] W[k,i].
                    # We want to minimize sum_k (u * V[j,k] W[k,i] - R[i,j,k])^2.
                    # This is a scalar least squares problem for each (i,j)!
                    # u = sum_k R[i,j,k] V[j,k] W[k,i] / sum_k (V[j,k] W[k,i])^2
                    
                    V_jk = V_r[:, :, r] # (m, p)
                    W_ki = W_r[:, :, r] # (p, n)
                    
                    # Numerator: sum_k R[i,j,k] V[j,k] W[k,i]
                    # R is (n, m, p). V is (m, p). W is (p, n).
                    # num[i,j] = sum_k R[i,j,k] V[j,k] W[k,i]
                    num = np.einsum('ijk,jk,ki->ij', residual, V_jk, W_ki)
                    
                    # Denominator: sum_k (V[j,k] W[k,i])^2
                    den = np.einsum('jk,jk,ki,ki->ij', V_jk, V_jk, W_ki, W_ki)
                    
                    # Update U_r[i,j]
                    mask = den > 1e-15
                    new_U_r = np.zeros((n, m), dtype=np.float64)
                    new_U_r[mask] = num[mask] / den[mask]
                    
                    # Update tensor_sum
                    old_contrib = U_r[:, :, r] * V_jk * W_ki
                    new_contrib = new_U_r * V_jk * W_ki
                    tensor_sum = tensor_sum - old_contrib + new_contrib
                    U_r[:, :, r] = new_U_r
                
                # Update V
                for r in range(R):
                    contrib_r = U_r[:, :, r] * V_r[:, :, r] * W_r[:, :, r]
                    residual = T - tensor_sum + contrib_r
                    
                    U_ij = U_r[:, :, r] # (n, m)
                    W_ki = W_r[:, :, r] # (p, n)
                    
                    # num[j,k] = sum_i R[i,j,k] U[i,j] W[k,i]
                    num = np.einsum('ijk,ij,ki->jk', residual, U_ij, W_ki)
                    
                    # den[j,k] = sum_i (U[i,j] W[k,i])^2
                    den = np.einsum('ij,ij,ki,ki->jk', U_ij, U_ij, W_ki, W_ki)
                    
                    mask = den > 1e-15
                    new_V_r = np.zeros((m, p), dtype=np.float64)
                    new_V_r[mask] = num[mask] / den[mask]
                    
                    old_contrib = U_ij * V_r[:, :, r] * W_ki
                    new_contrib = U_ij * new_V_r * W_ki
                    tensor_sum = tensor_sum - old_contrib + new_contrib
                    V_r[:, :, r] = new_V_r
                
                # Update W
                for r in range(R):
                    contrib_r = U_r[:, :, r] * V_r[:, :, r] * W_r[:, :, r]
                    residual = T - tensor_sum + contrib_r
                    
                    U_ij = U_r[:, :, r] # (n, m)
                    V_jk = V_r[:, :, r] # (m, p)
                    
                    # num[k,i] = sum_j R[i,j,k] U[i,j] V[j,k]
                    # Note: R is (n,m,p). We want index (k,i).
                    # R[i,j,k] -> R_transposed[k,i,j]?
                    # einsum: 'ijk,ij,jk->ki'
                    num = np.einsum('ijk,ij,jk->ki', residual, U_ij, V_jk)
                    
                    # den[k,i] = sum_j (U[i,j] V[j,k])^2
                    den = np.einsum('ij,ij,jk,jk->ki', U_ij, U_ij, V_jk, V_jk)
                    
                    mask = den > 1e-15
                    new_W_r = np.zeros((p, n), dtype=np.float64)
                    new_W_r[mask] = num[mask] / den[mask]
                    
                    old_contrib = U_ij * V_jk * W_r[:, :, r]
                    new_contrib = U_ij * V_jk * new_W_r
                    tensor_sum = tensor_sum - old_contrib + new_contrib
                    W_r[:, :, r] = new_W_r
                
                error = np.sum((tensor_sum - T) ** 2)
                if error < 1e-10:
                    # Check if solution is close to half-integers
                    U_flat = U_r.flatten()
                    V_flat = V_r.flatten()
                    W_flat = W_r.flatten()
                    
                    # Try rounding to nearest half-integer
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    # Verify rounded solution
                    U_check = U_rounded.reshape(n, m, R)
                    V_check = V_rounded.reshape(m, p, R)
                    W_check = W_rounded.reshape(p, n, R)
                    T_check = np.einsum('ijr,jkr,kir->ijk', U_check, V_check, W_check)
                    
                    if np.allclose(T_check, T, atol=1e-9):
                        best_rank = R
                        best_U = U_rounded.reshape(n*m, R).tolist()
                        best_V = V_rounded.reshape(m*p, R).tolist()
                        best_W = W_rounded.reshape(p*n, R).tolist()
                        return True
                    else:
                        # If not half-integer, maybe integer?
                        U_rounded_int = np.round(U_flat)
                        V_rounded_int = np.round(V_flat)
                        W_rounded_int = np.round(W_flat)
                        U_check = U_rounded_int.reshape(n, m, R)
                        V_check = V_rounded_int.reshape(m, p, R)
                        W_check = W_rounded_int.reshape(p, n, R)
                        T_check = np.einsum('ijr,jkr,kir->ijk', U_check, V_check, W_check)
                        if np.allclose(T_check, T, atol=1e-9):
                            best_rank = R
                            best_U = U_rounded_int.reshape(n*m, R).tolist()
                            best_V = V_rounded_int.reshape(m*p, R).tolist()
                            best_W = W_rounded_int.reshape(p*n, R).tolist()
                            return True
            
    # Try rank 119
    if try_rank(119):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If 119 fails, return schoolbook 120
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
