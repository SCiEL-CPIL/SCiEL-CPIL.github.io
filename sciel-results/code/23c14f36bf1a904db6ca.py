import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Seed solution (Rank 24) - schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)

    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def check_exact(U, V, W):
        # Verify exactness with integer/half-integer logic
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def polish_solution(U, V, W, max_iters=50, ftol=1e-12):
        """
        Refine a candidate solution using scipy.optimize.least_squares.
        This handles the non-convex optimization more robustly than ALS alone.
        """
        if time.time() > deadline - 5.0:
            return U, V, W

        # Flatten parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        size_U = U.size
        size_V = V.size
        
        def residuals(x):
            U_curr = x[:size_U].reshape(U.shape)
            V_curr = x[size_U:size_U+size_V].reshape(V.shape)
            W_curr = x[size_U+size_V:].reshape(W.shape)
            
            # Compute tensor values
            vals = np.sum(U_curr[u_idxs, :] * V_curr[v_idxs, :] * W_curr[w_idxs, :], axis=1)
            return vals - 1.0

        try:
            res = opt.least_squares(residuals, x0, method='lm', max_nfev=1000, ftol=ftol, xtol=ftol, gtol=ftol)
            x_opt = res.x
            U_opt = x_opt[:size_U].reshape(U.shape)
            V_opt = x_opt[size_U:size_U+size_V].reshape(V.shape)
            W_opt = x_opt[size_U+size_V:].reshape(W.shape)
            
            # Check if the polished solution is better
            if res.cost < error_function(U, V, W):
                return U_opt, V_opt, W_opt
        except Exception:
            pass
        return U, V, W

    def try_round(U, V, W):
        """
        Attempt to round the continuous solution to integer/half-integer values.
        We try multiple scaling factors to handle the inherent scaling invariance.
        """
        # The decomposition is invariant under U->cU, V->V/c, W->W for any c != 0.
        # We can adjust the scaling of one matrix to make rounding easier.
        # Common strategy: scale U to have unit norm columns, then round V and W?
        # Or just try a range of scales for the whole system?
        
        # Let's try scaling U by a factor s, and see if V*s and W/s (or similar) round well.
        # Actually, the product U*V*W is invariant.
        # If we scale U by s, we must scale V by 1/s to keep U*V constant? No, W is separate.
        # U[i,r] V[j,r] W[k,r].
        # If U' = s U, V' = V/s, W' = W, then U'V'W' = U V W.
        
        # Let's try a set of scaling factors for U (and inverse for V).
        scales = [0.5, 1.0, 2.0, 0.1, 10.0]
        
        best_rounded = None
        best_err = float('inf')

        for s in scales:
            U_s = U * s
            V_s = V / s
            W_s = W
            
            # Round to nearest 0.5
            U_r = np.round(U_s * 2) / 2
            V_r = np.round(V_s * 2) / 2
            W_r = np.round(W_s * 2) / 2
            
            if check_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
            
            # Round to nearest integer
            U_r = np.round(U_s)
            V_r = np.round(V_s)
            W_r = np.round(W_s)
            
            if check_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
                
            # Track the best "almost" exact solution for potential further polishing
            err = error_function(U_r, V_r, W_r)
            if err < best_err:
                best_err = err
                best_rounded = (U_r, V_r, W_r)
                
        return best_rounded

    def try_rank_als(r, max_iters=20, num_restarts=50, seed_base=42):
        if time.time() > deadline - 10.0:
            return None
            
        best_candidate = None
        best_candidate_err = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            np.random.seed(seed_base + restart)
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Normalize initial guesses
            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                
                # Vectorized ALS Update
                # For U: minimize || U[u,:] * (V[v,:] o W[w,:]) - 1 ||^2
                # This is equivalent to solving (V[v,:] o W[w,:])^T (V[v,:] o W[w,:]) U[u,:] = (V[v,:] o W[w,:])^T 1
                # We can compute the Gram matrices efficiently.
                
                # Gram_U[r, r] = sum_c V[v_c, r] * W[w_c, r] * V[v_c, r'] * W[w_c, r']
                # This can be done by:
                # V_v = V[v_idxs, :] shape (C, r)
                # W_w = W[w_idxs, :] shape (C, r)
                # outer_products = V_v[:, :, None] * W_w[:, None, :] shape (C, r, r)
                # Gram_U = outer_products.sum(axis=0)
                
                V_v = V[v_idxs, :]
                W_w = W[w_idxs, :]
                U_u = U[u_idxs, :]
                
                # Compute Gram matrices
                # Note: np.einsum might be faster, but let's use broadcasting
                # Gram_U = (V_v[:, :, None] * W_w[:, None, :]).sum(axis=0)
                # RHS_U = (V_v[:, :, None] * W_w[:, None, :]).sum(axis=0) @ 1? No.
                # RHS_U[r] = sum_c V[v_c, r] * W[w_c, r]
                
                prod_VW = V_v * W_w # (C, r)
                Gram_U = prod_VW.T @ prod_VW # (r, r)
                RHS_U = prod_VW.sum(axis=0) # (r,)
                
                prod_UW = U_u * W_w
                Gram_V = prod_UW.T @ prod_UW
                RHS_V = prod_UW.sum(axis=0)
                
                prod_UV = U_u * V_v
                Gram_W = prod_UV.T @ prod_UV
                RHS_W = prod_UV.sum(axis=0)
                
                # Solve
                try:
                    # Add small regularization for stability
                    reg = 1e-8
                    U_flat = np.linalg.solve(Gram_U + reg * np.eye(r), RHS_U)
                    V_flat = np.linalg.solve(Gram_V + reg * np.eye(r), RHS_V)
                    W_flat = np.linalg.solve(Gram_W + reg * np.eye(r), RHS_W)
                except np.linalg.LinAlgError:
                    continue
                
                U = U_flat.reshape(n * m, r)
                V = V_flat.reshape(m * p, r)
                W = W_flat.reshape(p * n, r)
                
                # Normalize to prevent drift
                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)
                
                err = error_function(U, V, W)
                
                if err < best_candidate_err:
                    best_candidate_err = err
                    best_candidate = (U.copy(), V.copy(), W.copy())
                    
                if err < 1e-10:
                    break
                
                if err > prev_err * 1.05:
                    break
                prev_err = err

        if best_candidate is None:
            return None

        U_c, V_c, W_c = best_candidate
        
        # Polish the best candidate
        U_p, V_p, W_p = polish_solution(U_c, V_c, W_c)
        
        # Try rounding
        rounded = try_round(U_p, V_p, W_p)
        if rounded:
            U_r, V_r, W_r = rounded
            if check_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
                
        # If rounding failed, check if the continuous solution is "close enough" to be considered for next steps?
        # For this task, we need exactness. So we return None if not exact.
        # However, we can return the polished solution if we want to allow further polishing in a higher rank?
        # No, the rank is fixed.
        
        return None

    # Try lower ranks
    ranks_to_try = [23, 22, 21, 20]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Allocate more time/restarts for higher ranks
        if r == 23:
            sol = try_rank_als(r, max_iters=30, num_restarts=100, seed_base=r*100)
        elif r == 22:
            sol = try_rank_als(r, max_iters=30, num_restarts=50, seed_base=r*100)
        else:
            sol = try_rank_als(r, max_iters=20, num_restarts=20, seed_base=r*100)
            
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    # If we didn't find a better rank, return the seed
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
