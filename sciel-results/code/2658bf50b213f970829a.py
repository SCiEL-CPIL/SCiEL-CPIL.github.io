import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
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
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
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

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
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

    def greedy_elimination(U, V, W, max_rounds=30):
        """
        Greedy column elimination: try to remove each column by checking if
        the residual can be expressed as a linear combination of remaining columns.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        active = list(range(U.shape[1]))
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            eliminated = False
            for r in active[:]:
                if time.time() > deadline - 5:
                    break
                
                # Remove column r
                others = [s for s in active if s != r]
                if len(others) < 1:
                    break
                
                # Residual tensor after removing column r
                recon_others = np.einsum('ar,br,cr->abc', U[:, others], V[:, others], W[:, others])
                residual = T - recon_others
                
                # Check if residual can be expressed as sum of new rank-1 terms
                # from the remaining columns. This is a linear system:
                # For each (a,b,c), residual[a,b,c] = sum_s alpha_s * U[a,s] * V[b,s] * W[c,s]
                # This is nonlinear in the coefficients if we allow U,V,W to change.
                # Instead, we fix U,V,W of remaining columns and ask: is residual
                # in the span of {U[:,s] ⊗ V[:,s] ⊗ W[:,s] : s in others}?
                
                # Flatten residual and build design matrix
                res_flat = residual.flatten()
                # Design matrix: each column is U[:,s] ⊗ V[:,s] ⊗ W[:,s] flattened
                A_mat = np.zeros((res_flat.size, len(others)))
                for idx, s in enumerate(others):
                    A_mat[:, idx] = np.kron(np.kron(U[:, s], V[:, s]), W[:, s])
                
                # Solve least squares
                try:
                    coeffs, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, res_flat, rcond=1e-10)
                except np.linalg.LinAlgError:
                    continue
                
                # Check if the solution is good
                recon_check = A_mat @ coeffs
                if np.allclose(recon_check, res_flat, atol=1e-9):
                    # Column r is redundant! Remove it.
                    # But wait: the coefficients alpha_s modify the remaining columns.
                    # We need to update U,V,W for the remaining columns.
                    # This is tricky because alpha_s * U[:,s] ⊗ V[:,s] ⊗ W[:,s]
                    # can't generally be decomposed as a single rank-1 term unless
                    # we allow the coefficients to be absorbed.
                    
                    # Actually, the correct interpretation: if residual is in the span,
                    # then T = sum_{s in others} (1 + alpha_s) * U[:,s] ⊗ V[:,s] ⊗ W[:,s]
                    # only if the coefficients can be absorbed into one factor.
                    # This is NOT generally possible.
                    
                    # So the greedy elimination only works if we can find a single
                    # column that is EXACTLY a linear combination in a way that
                    # preserves the rank-1 structure. This is very restrictive.
                    
                    # Alternative: just check if the column is zero or if removing it
                    # and re-optimizing works. For now, skip this approach.
                    pass
                
                # Try a different approach: check if column r is a linear combination
                # of others in a way that preserves structure. This is hard.
                # Instead, let's try a simpler heuristic: check if the column has
                # very small norm (numerically redundant).
                col_norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
                if col_norm < 1e-10:
                    active.remove(r)
                    eliminated = True
                    break
            
            if not eliminated:
                break
        
        if not active:
            return None, None, None
        return U[:, active], V[:, active], W[:, active]

    def try_lower_rank_optimize(U_init, V_init, W_init, target_rank):
        """
        Try to find a rank-target_rank decomposition by optimizing from an
        initialization derived from the rank-50 solution.
        """
        if time.time() > deadline - 5:
            return None, None, None
        
        # Initialization: use first target_rank columns
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        def loss_and_grad(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            error = recon - T
            loss = np.sum(error**2)
            
            dU = np.einsum('abc,br,cr->ar', error, V_flat, W_flat)
            dV = np.einsum('abc,ar,cr->br', error, U_flat, W_flat)
            dW = np.einsum('abc,ar,br->cr', error, U_flat, V_flat)
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return loss, grad

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(loss_and_grad, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e-8:
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                if verify_decomp(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
                    
        except Exception:
            pass
        
        return None, None, None

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U50, V50, W50
    best_rank = 50

    # Try greedy elimination first (fast)
    if time.time() < deadline - 20:
        U_g, V_g, W_g = greedy_elimination(U50, V50, W50)
        if U_g is not None and U_g.shape[1] < best_rank:
            if verify_decomp(U_g, V_g, W_g):
                best_U, best_V, best_W = U_g, V_g, W_g
                best_rank = U_g.shape[1]

    # Try optimization-based rank reduction
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, 44, -1):
            if time.time() > deadline - 5:
                break
            
            # Try multiple initializations
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                
                if attempt == 0:
                    # First target_rank columns
                    U_init = U50[:, :target_rank]
                    V_init = V50[:, :target_rank]
                    W_init = W50[:, :target_rank]
                elif attempt == 1:
                    # Random subset
                    np.random.seed(42 + target_rank)
                    idx = np.random.choice(50, target_rank, replace=False)
                    U_init = U50[:, idx]
                    V_init = V50[:, idx]
                    W_init = W50[:, idx]
                else:
                    # Last target_rank columns
                    U_init = U50[:, 50-target_rank:]
                    V_init = V50[:, 50-target_rank:]
                    W_init = W50[:, 50-target_rank:]
                
                U_opt, V_opt, W_opt = try_lower_rank_optimize(U_init, V_init, W_init, target_rank)
                if U_opt is not None:
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = target_rank
                    break  # Found solution for this rank, try lower

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
