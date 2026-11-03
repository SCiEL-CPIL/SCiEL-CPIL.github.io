import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a, b, c, d, k, i] = delta_b_c * delta_a_i * delta_d_k
    # Shape: (n*m, m*p, p*n) -> (8, 20, 10)
    # T[a*m+b, c*p+d, k*n+i] = 1 if b==c and a==i and d==k, else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                a = i  # delta_a_i
                d = k  # delta_d_k
                c = b  # delta_b_c
                T[a*m+b, c*p+d, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill

    def verify_solution_copy(U, V, W, tol=1e-9):
        T_copy = T.copy()
        R = U.shape[1]
        for mu in range(R):
            T_copy -= np.einsum('a,b,c->abc', U[:,mu], V[:,mu], W[:,mu])
        return np.max(np.abs(T_copy)) < tol

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def check_and_snap(U, V, W, tol=1e-9):
        U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
        T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.max(np.abs(T_rec_snap - T)) < tol:
            return U_snap, V_snap, W_snap
        return None

    # --- Phase 1: Construct known Strassen-based rank-36 solution ---
    def get_strassen_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        # Verify and return
        res = check_and_snap(U, V, W)
        if res is not None:
            return res
        return (U, V, W)

    U36, V36, W36 = get_strassen_rank36()
    
    best_rank = 36
    best_result = (U36, V36, W36)

    # --- Phase 2: Optimization to find lower rank ---
    def residual_vector_vectorized(x, R):
        size_U = nm * R
        size_V = mp * R
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U+size_V].reshape(mp, R)
        W = x[size_U+size_V:].reshape(pn, R)
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()
    
    def gradient_residual_vectorized(x, R):
        size_U = nm * R
        size_V = mp * R
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U+size_V].reshape(mp, R)
        W = x[size_U+size_V:].reshape(pn, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        residual = T_rec - T
        
        grad_U = np.einsum('abc,bm,cm->am', residual, V, W, optimize=True)
        grad_V = np.einsum('abc,am,cm->bm', residual, U, W, optimize=True)
        grad_W = np.einsum('abc,am,bm->cm', residual, U, V, optimize=True)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    def try_optimize(U_init, V_init, W_init, R, maxiter=2000, maxfun=4000):
        size_U = nm * R
        size_V = mp * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient_residual_vectorized(x, R),
                maxiter=maxiter,
                maxfun=maxfun,
                options={'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 80}
            )
            
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(nm, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
            W_opt = x_opt[size_U+size_V:].reshape(pn, R)
            
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T))
            
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf
    
    def polish(U_init, V_init, W_init, R):
        size_U = nm * R
        size_V = mp * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient_residual_vectorized(x, R),
                maxiter=10000,
                maxfun=20000,
                options={'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(nm, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
            W_opt = x_opt[size_U+size_V:].reshape(pn, R)
            
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T))
            
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf

    # --- Phase 3: Sparse Random Linear Probing (Inspiration Strategy) ---
    def sparse_probing_search(R, n_candidates=500, time_budget=5.0):
        """
        Generate random rank-1 tensors with sparse integer/half-integer entries.
        Use LASSO or similar sparse regression to find a sparse combination.
        """
        if time.time() - start_time > time_limit - 10:
            return None
        
        # Generate candidates
        # Each candidate is a rank-1 tensor defined by u, v, w vectors
        # We want to find coefficients c_mu such that sum c_mu * (u_mu \otimes v_mu \otimes w_mu) = T
        # This is equivalent to finding U, V, W with R columns.
        # Here we generate random sparse U, V, W and solve for scaling?
        # No, the rank-1 terms are the columns.
        # Let's generate R random sparse rank-1 components and see if they span T.
        # Actually, LASSO on the flattened tensor with a dictionary of random rank-1 tensors.
        
        # Dictionary size N_dict
        N_dict = min(200, int(n_candidates))
        
        # Generate random sparse rank-1 vectors
        # U shape (nm, N_dict), V shape (mp, N_dict), W shape (pn, N_dict)
        # Each column is a sparse vector with few non-zeros.
        
        U_dict = np.zeros((nm, N_dict))
        V_dict = np.zeros((mp, N_dict))
        W_dict = np.zeros((pn, N_dict))
        
        for mu in range(N_dict):
            # Random sparse pattern
            nnz_u = np.random.choice([2, 3, 4])
            nnz_v = np.random.choice([2, 3, 4])
            nnz_w = np.random.choice([2, 3, 4])
            
            idx_u = np.random.choice(nm, size=nnz_u, replace=False)
            idx_v = np.random.choice(mp, size=nnz_v, replace=False)
            idx_w = np.random.choice(pn, size=nnz_w, replace=False)
            
            U_dict[idx_u, mu] = np.random.choice([-1, 1], size=nnz_u)
            V_dict[idx_v, mu] = np.random.choice([-1, 1], size=nnz_v)
            W_dict[idx_w, mu] = np.random.choice([-1, 1], size=nnz_w)
        
        # Build design matrix X where each column is the flattened tensor product
        # X[:, mu] = vec(u_mu \otimes v_mu \otimes w_mu)
        # This is huge: nm*mp*pn = 1600. N_dict=200.
        # X shape (1600, 200)
        
        X = np.zeros((nm*mp*pn, N_dict))
        for mu in range(N_dict):
            tensor_mu = np.einsum('a,b,c->abc', U_dict[:,mu], V_dict[:,mu], W_dict[:,mu])
            X[:, mu] = tensor_mu.flatten()
        
        y = T.flatten()
        
        # Solve min ||X c - y||^2 + lambda ||c||_1
        # We can use scipy.optimize.lsq_linear or similar, but LASSO is better for sparsity.
        # scipy doesn't have a built-in LASSO in standard optimize.
        # We can use a simple iterative soft-thresholding or just least squares and check sparsity.
        # Given small dimensions, let's try least squares first.
        
        try:
            # Solve X c = y
            c, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            
            # Check if the solution is sparse and exact
            # If c is sparse, we can extract the non-zero components.
            # But we need to reconstruct U, V, W from the selected dictionary entries.
            
            # Threshold c
            c_sparse = c.copy()
            c_sparse[np.abs(c_sparse) < 1e-6] = 0
            
            if np.allclose(X @ c_sparse, y, atol=1e-9):
                # Extract non-zero indices
                non_zero_idx = np.where(np.abs(c_sparse) > 1e-6)[0]
                
                if len(non_zero_idx) <= R:
                    # We found a solution with rank <= R using the dictionary
                    # But the entries in U, V, W are scaled by c?
                    # No, the tensor is sum c_mu * (u_mu \otimes v_mu \otimes w_mu).
                    # This can be rewritten as sum (sqrt(c_mu) u_mu) \otimes (sqrt(c_mu) v_mu) \otimes (1/sqrt(c_mu) w_mu) ? No.
                    # It is sum (c_mu^{1/3} u_mu) \otimes (c_mu^{1/3} v_mu) \otimes (c_mu^{1/3} w_mu).
                    # This scaling might break integer/half-integer constraints.
                    # However, if we just use the columns directly:
                    # U_new[:, i] = c_mu * U_dict[:, mu] ? No, that changes the tensor.
                    # Tensor = sum c_mu * T_mu.
                    # If we define U_col = c_mu * u_mu, V_col = v_mu, W_col = w_mu, then U_col \otimes V_col \otimes W_col = c_mu * T_mu.
                    # So we can absorb c_mu into one of the factors.
                    
                    U_new = np.zeros((nm, len(non_zero_idx)))
                    V_new = np.zeros((mp, len(non_zero_idx)))
                    W_new = np.zeros((pn, len(non_zero_idx)))
                    
                    for i, mu in enumerate(non_zero_idx):
                        U_new[:, i] = c_sparse[mu] * U_dict[:, mu]
                        V_new[:, i] = V_dict[:, mu]
                        W_new[:, i] = W_dict[:, mu]
                    
                    # Verify
                    res = check_and_snap(U_new, V_new, W_new)
                    if res is not None:
                        return res
            
        except Exception:
            pass
        
        return None

    # Main Search Loop
    for R in range(35, 23, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        remaining = time_limit - 10 - (time.time() - start_time)
        if remaining < 5:
            break
        
        found = False
        
        # Attempt 1: Truncate rank-36 solution
        if R <= 36:
            U_init = U36[:, :R].copy()
            V_init = V36[:, :R].copy()
            W_init = W36[:, :R].copy()
            
            U_opt, V_opt, W_opt, max_res = try_optimize(U_init, V_init, W_init, R)
            if U_opt is not None and max_res < 1e-6:
                U_pol, V_pol, W_pol, max_res_pol = polish(U_opt, V_opt, W_opt, R)
                if U_pol is not None and max_res_pol < 1e-8:
                    res = check_and_snap(U_pol, V_pol, W_pol)
                    if res is not None:
                        best_rank = R
                        best_result = res
                        found = True
                        break

        # Attempt 2: Sparse Probing
        if not found:
            res = sparse_probing_search(R)
            if res is not None:
                best_rank = R
                best_result = res
                found = True
                break

        # Attempt 3: Random Initializations
        for attempt in range(10):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Various random inits
            if attempt == 0:
                U_init = np.random.normal(0, 1.0, (nm, R))
                V_init = np.random.normal(0, 1.0, (mp, R))
                W_init = np.random.normal(0, 1.0, (pn, R))
            elif attempt == 1:
                U_init = np.random.choice([-1, 0, 1], (nm, R)).astype(float)
                V_init = np.random.choice([-1, 0, 1], (mp, R)).astype(float)
                W_init = np.random.choice([-1, 0, 1], (pn, R)).astype(float)
            else:
                scale = 0.5 + 0.1 * attempt
                U_init = np.random.normal(0, scale, (nm, R))
                V_init = np.random.normal(0, scale, (mp, R))
                W_init = np.random.normal(0, scale, (pn, R))
            
            U_opt, V_opt, W_opt, max_res = try_optimize(U_init, V_init, W_init, R, maxiter=1000, maxfun=2000)
            
            if U_opt is not None and max_res < 1e-6:
                U_pol, V_pol, W_pol, max_res_pol = polish(U_opt, V_opt, W_opt, R)
                if U_pol is not None and max_res_pol < 1e-8:
                    res = check_and_snap(U_pol, V_pol, W_pol)
                    if res is not None:
                        best_rank = R
                        best_result = res
                        found = True
                        break
        
        if found:
            break

    # Final Fallback
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Should not reach here if rank-36 is valid
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
