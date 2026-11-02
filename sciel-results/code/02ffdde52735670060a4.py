import time
import sys
import numpy as np

DIMENSIONS = [2, 2, 11]
DEADLINE = 150.0
SAFETY = 5.0  # seconds to leave before hard kill

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # --- Guaranteed rank-39 construction (Strassen blocks + schoolbook tail) ---
    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
            w_c00 = k0 * n; w_c10 = k0 * n + 1
            w_c01 = k1 * n; w_c11 = k1 * n + 1
            # m1
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            # m2
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            # m3
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            # m4
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            # m5
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            # m6
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            # m7
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        # single column k2 = 10, schoolbook 4
        k2 = 10
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n; w_c12 = k2 * n + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}

    best = build_rank39()

    # --- Verification helpers ---
    def verify(U, V, W):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T shape: (n*m, m*p, p*n)
        T = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        
        # Construct target tensor M of shape (n*m, m*p, p*n)
        M = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0
        
        return np.allclose(T, M, atol=1e-9, rtol=0.0)

    def round_halfint(mat):
        return np.round(np.asarray(mat, dtype=float) * 2.0) / 2.0

    # --- Null-space projection approach ---
    # For each fixed a (input index into U), the constraint is:
    #   sum_r U[a,r] * V[:,r] = W[:,a]  (as a vector over the W-index)
    # Actually the full tensor constraint is:
    #   T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = M[a,b,c]
    # For fixed a, this gives: sum_r U[a,r] * (V[:,r] outer W[:,r]) = M[a,:,:]
    # This is bilinear in (V,W) for fixed U.
    #
    # Alternative: fix the "shape" of which entries are nonzero (support pattern),
    # then solve the linear system for the values.
    #
    # Better approach: For a given rank r, we have r columns. Each column r contributes
    # a rank-1 tensor U_col_r ⊗ V_col_r ⊗ W_col_r.
    # The constraint is: sum_r U_col_r ⊗ V_col_r ⊗ W_col_r = M
    #
    # Key insight: For fixed V and W (or fixed U), the problem is linear in the other factors.
    # Specifically, if we fix V and W, then for each (b,c): sum_r V[b,r]*W[c,r]*U[a,r] = M[a,b,c]
    # This is linear in U. Similarly for V given U,W, and for W given U,V.
    #
    # Strategy: Use the null-space of the "linearized" system.
    # For a candidate rank r, parameterize:
    #   x = vec(U) in R^{4*r}, y = vec(V) in R^{22*r}, z = vec(W) in R^{22*r}
    # The tensor equation is trilinear, but if we fix two, it's linear in the third.
    #
    # Practical approach: Start from rank-39, remove columns one at a time,
    # and for each removal, try to re-solve the remaining system with the freed degrees.
    #
    # More targeted: For rank r < 39, we need to find U(4×r), V(22×r), W(22×r)
    # such that einsum gives M. Fix the support pattern (which entries are nonzero)
    # and solve the resulting system.
    #
    # Implementation: Use the fact that for each "block" of the multiplication,
    # we can try different rank-1 decompositions.

    def try_nullspace_search(max_rank, min_rank, budget_s):
        """Try to find lower-rank solutions using structured search over support patterns."""
        nonlocal best
        if time.time() - t_start > DEADLINE - SAFETY:
            return

        t_search_start = time.time()
        rng = np.random.default_rng(123)

        # Build the target tensor
        M = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0

        # For a given rank r, the number of free parameters is r*(4+22+22) = 48r
        # The number of constraints is 4*22*22 = 1936 (but many are zero)
        # Nonzero entries of M: 4*11 = 44 (one for each (i,j,k) triple)
        # So we have 44 equations in 48r unknowns (with trilinear structure).
        #
        # For r=1: 48 unknowns, 44 equations -> potentially solvable but likely no solution
        # For r=2: 96 unknowns, 44 equations -> overdetermined in a good way
        #
        # The key challenge is the trilinear structure. Let's use an alternating approach:
        # Fix V,W -> solve for U (linear). Fix U,W -> solve for V. Fix U,V -> solve for W.

        def alt_minimize(U_init, V_init, W_init, n_iter=50, tol=1e-10):
            """Alternating least squares for tensor decomposition."""
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            for it in range(n_iter):
                if time.time() - t_search_start > budget_s:
                    return None
                # Solve for U: for each a, M[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
                # This is: for each a, M[a].reshape(-1) = sum_r U[a,r] * (V[:,r]*W[:,r]).reshape(-1)
                # Build matrix A_U where A_U[a, r] doesn't work directly because it's per-a.
                # Actually: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: M[a] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # So M[a].reshape(1,-1) = U[a] @ (V⊗W) where (V⊗W) is 22*22 × r
                # (V⊗W)[(b,c), r] = V[b,r]*W[c,r]
                VW = np.einsum('br,cr->bcr', V, W).reshape(m*p, -1)  # (22, r)
                # For each a: M[a].reshape(-1) = U[a] @ VW
                # Stacking: M.reshape(4, -1) = U @ VW
                M_flat = M.reshape(n*m, m*p*p*n)
                # Actually M[a] has shape (m*p, p*n), flatten to (m*p*p*n,)
                M_2d = M.reshape(n*m, -1)  # (4, 484)
                # Solve U @ VW = M_2d
                # U is (4, r), VW is (484, r)
                # U = M_2d @ pinv(VW)
                if r > 0:
                    VW_pinv = np.linalg.pinv(VW)
                    U_new = M_2d @ VW_pinv
                    U = U_new
                # Solve for V: M[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed b: M[:,:,b] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
                # M[:,:,b].reshape(-1) = V[b] @ (U⊗W) where (U⊗W)[(a,c),r] = U[a,r]*W[c,r]
                UW = np.einsum('ar,cr->acr', U, W).reshape(n*m, -1)  # (4, r)
                # M[:,:,b] for all b: M.reshape(m*p, n*m, p*n) -> transpose to (m*p, n*m*p*n)
                M_b = M.reshape(m*p, n*m, p*n).transpose(0, 1, 2).reshape(m*p, -1)  # (22, 88)
                # Wait, M[a,b,c], for fixed b: M[:,b,:].reshape(-1) = V[b] @ UW
                # UW has shape (n*m, r) = (4, r), so V[b] @ UW gives (4,) not (4*p*n,)
                # Let me redo: M[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Fix b: M[:,b,:].reshape(-1) of shape (4*22,) = sum_r V[b,r] * (U[:,r]*W[:,r]).reshape(-1)
                # (U[:,r]*W[:,r]) has shape (4, 22), flatten to (88,)
                # So M[:,b,:].reshape(-1) = V[b] @ (U⊗W) where (U⊗W)[(a,c),r] = U[a,r]*W[c,r]
                UW_full = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, -1)  # (88, r)
                M_b_flat = M.transpose(1, 0, 2).reshape(m*p, -1)  # (22, 88)
                if r > 0:
                    UW_pinv = np.linalg.pinv(UW_full)
                    V_new = M_b_flat @ UW_pinv
                    V = V_new
                # Solve for W: M[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Fix c: M[:,:,c].reshape(-1) = W[c] @ (U⊗V) where (U⊗V)[(a,b),r] = U[a,r]*V[b,r]
                UV_full = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, -1)  # (88, r)
                M_c_flat = M.transpose(2, 0, 1).reshape(p*n, -1)  # (22, 88)
                if r > 0:
                    UV_pinv = np.linalg.pinv(UV_full)
                    W_new = M_c_flat @ UV_pinv
                    W = W_new
            # Check convergence
            T = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T, M, atol=1e-8, rtol=0):
                return (U, V, W)
            return None

        # Try ranks from max_rank down
        for rank in range(max_rank, min_rank - 1, -1):
            if time.time() - t_search_start > budget_s:
                break
            if time.time() - t_start > DEADLINE - SAFETY:
                break
            
            # Initialize from rank-39 solution (take first 'rank' columns)
            U0 = np.asarray(best["U"], dtype=float)
            V0 = np.asarray(best["V"], dtype=float)
            W0 = np.asarray(best["W"], dtype=float)
            
            n_attempts = 5
            for attempt in range(n_attempts):
                if time.time() - t_search_start > budget_s:
                    break
                if time.time() - t_start > DEADLINE - SAFETY:
                    break
                
                # Different initializations
                if attempt == 0:
                    # Take first rank columns
                    U_init = U0[:, :rank].copy()
                    V_init = V0[:, :rank].copy()
                    W_init = W0[:, :rank].copy()
                elif attempt == 1:
                    # Random subset
                    idx = rng.choice(U0.shape[1], size=rank, replace=False)
                    U_init = U0[:, idx].copy()
                    V_init = V0[:, idx].copy()
                    W_init = W0[:, idx].copy()
                elif attempt == 2:
                    # Random initialization with small values
                    U_init = rng.normal(0, 0.5, (n*m, rank))
                    V_init = rng.normal(0, 0.5, (m*p, rank))
                    W_init = rng.normal(0, 0.5, (p*n, rank))
                elif attempt == 3:
                    # Perturbed first rank columns
                    U_init = U0[:, :rank].copy() + rng.normal(0, 0.1, (n*m, rank))
                    V_init = V0[:, :rank].copy() + rng.normal(0, 0.1, (m*p, rank))
                    W_init = W0[:, :rank].copy() + rng.normal(0, 0.1, (p*n, rank))
                else:
                    # Structured: use Strassen-like patterns
                    U_init = np.zeros((n*m, rank))
                    V_init = np.zeros((m*p, rank))
                    W_init = np.zeros((p*n, rank))
                    for r_idx in range(rank):
                        # Random support pattern
                        u_idx = rng.choice(n*m, size=rng.integers(1, 4), replace=False)
                        v_idx = rng.choice(m*p, size=rng.integers(1, 4), replace=False)
                        w_idx = rng.choice(p*n, size=rng.integers(1, 4), replace=False)
                        U_init[u_idx, r_idx] = rng.choice([-1, 1])
                        V_init[v_idx, r_idx] = rng.choice([-1, 1])
                        W_init[w_idx, r_idx] = rng.choice([-1, 1])
                
                result = alt_minimize(U_init, V_init, W_init, n_iter=100, tol=1e-10)
                if result is not None:
                    U_sol, V_sol, W_sol = result
                    # Round to half-integers
                    U_r = round_halfint(U_sol)
                    V_r = round_halfint(V_sol)
                    W_r = round_halfint(W_sol)
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        return True
                    # Also try integer rounding
                    U_i = np.round(U_sol)
                    V_i = np.round(V_sol)
                    W_i = np.round(W_sol)
                    if verify(U_i.tolist(), V_i.tolist(), W_i.tolist()):
                        best = {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                        return True
        return False

    # Try null-space / alternating search for rank 38 down to 34
    if time.time() - t_start < DEADLINE - SAFETY - 3:
        try_nullspace_search(38, 34, 10.0)

    # Fallback: original perturbation search if time remains
    def try_improve(max_rank, min_rank, budget_s):
        nonlocal best
        U0 = np.asarray(best["U"], dtype=float)
        V0 = np.asarray(best["V"], dtype=float)
        W0 = np.asarray(best["W"], dtype=float)
        rng = np.random.default_rng(42)

        for rank in range(max_rank, min_rank - 1, -1):
            if time.time() - t_start > DEADLINE - SAFETY:
                break
            sub_budget = min(10.0, max(1.0, (DEADLINE - SAFETY) - (time.time() - t_start)))
            t_rank_start = time.time()
            n_cols = U0.shape[1]
            for attempt in range(8):
                if time.time() - t_rank_start > sub_budget:
                    break
                if time.time() - t_start > DEADLINE - SAFETY:
                    break
                idx = rng.integers(0, n_cols, size=rank)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
                U = U + rng.normal(0, 0.1, U.shape)
                V = V + rng.normal(0, 0.1, V.shape)
                W = W + rng.normal(0, 0.1, W.shape)
                U = round_halfint(U)
                V = round_halfint(V)
                W = round_halfint(W)
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return True
        return False

    if time.time() - t_start < DEADLINE - SAFETY - 2:
        try_improve(38, 35, 5.0)

    return best
