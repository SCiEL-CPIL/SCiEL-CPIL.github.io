"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # The known construction: treat each output row i independently.
    # For a fixed row i, the subproblem is 1 x m x p matrix multiplication.
    # The tensor is T_{j,k} = 1 for all j,k.
    # Rank-m decomposition: for r in 0..m-1, set
    #   U_{j,r} = delta_{j,r}, V_{j,k,r} = 1, W_{k,r} = 1
    # For the full n x m x p case:
    #   For each (i, r) where r in 0..m-1:
    #     column c = i*m + r
    #     U: has 1 at position (i*m + r)  i.e. row index i*m+r in U
    #     V: has 1 at all positions (r*p + k) for k=0..p-1  (row index r*p+k)
    #     W: has 1 at all positions (k*n + i) for k=0..p-1  (row index k*n+i)
    #
    # Total rank = n * m
    def structured_construction():
        r = n * m
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for r_idx in range(m):
                c = i * m + r_idx
                # U: only row i*m + r_idx is non-zero
                U[i * m + r_idx, c] = 1.0
                # V: rows r_idx*p + k for all k
                for k in range(p):
                    V[r_idx * p + k, c] = 1.0
                # W: rows k*n + i for all k
                for k in range(p):
                    W[k * n + i, c] = 1.0
        return U, V, W

    U_s, V_s, W_s = structured_construction()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s

    def verify(U, V, W, tol=1e-10):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    if verify(best_U, best_V, best_W):
        # Found valid structured decomposition
        pass
    else:
        # Fallback to schoolbook
        U_s, V_s, W_s = schoolbook()
        best_rank = U_s.shape[1]
        best_U, best_V, best_W = U_s, V_s, W_s

    # Try to beat the structured rank using optimization
    deadline = time.time() + 140

    def compute_residual(U, V, W):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(S - T)

    def fast_als(U, V, W, max_iters=100, tol=1e-8):
        r = U.shape[1]
        if r == 0:
            return None

        Ur = U.copy()
        Vr = V.copy()
        Wr = W.copy()

        prev_res = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None

            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = T - term
                # Update U: S_minus_mat[a, (b,c)] contracted with V*b*W*c
                S_minus_mat = S_minus.reshape(nm, mp * pn)
                vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
                vw_norm_sq = np.dot(vw, vw)
                if vw_norm_sq > 1e-15:
                    Ur[:, r_idx] = S_minus_mat @ vw / vw_norm_sq

            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = T - term
                S_minus_mat = S_minus.reshape(mp, nm * pn)
                uw = np.outer(Ur[:, r_idx], Wr[:, r_idx]).flatten()
                uw_norm_sq = np.dot(uw, uw)
                if uw_norm_sq > 1e-15:
                    Vr[:, r_idx] = S_minus_mat @ uw / uw_norm_sq

            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = T - term
                S_minus_mat = S_minus.reshape(pn, nm * mp)
                uv = np.outer(Ur[:, r_idx], Vr[:, r_idx]).flatten()
                uv_norm_sq = np.dot(uv, uv)
                if uv_norm_sq > 1e-15:
                    Wr[:, r_idx] = S_minus_mat @ uv / uv_norm_sq

            if it % 5 == 0 or it == max_iters - 1:
                res = compute_residual(Ur, Vr, Wr)
                if res < prev_res:
                    prev_res = res
                if res < tol:
                    if verify(Ur, Vr, Wr):
                        return Ur, Vr, Wr
                    return None

        # Final check
        res = compute_residual(Ur, Vr, Wr)
        if res < 1e-8 and verify(Ur, Vr, Wr):
            return Ur, Vr, Wr
        return None

    def try_rank(rank, n_attempts=3, max_iters=80):
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break

            # Start from the structured solution and add perturbations for extra columns
            base = n * m
            if rank <= base:
                # For rank <= base, just use the structured one truncated
                U_init = best_U[:, :rank]
                V_init = best_V[:, :rank]
                W_init = best_W[:, :rank]
            else:
                U_init = best_U.copy()
                V_init = best_V.copy()
                W_init = best_W.copy()
                # Add random columns
                extra = rank - base
                U_init = np.hstack([best_U, np.random.randn(nm, extra) * 0.1])
                V_init = np.hstack([best_V, np.random.randn(mp, extra) * 0.1])
                W_init = np.hstack([best_W, np.random.randn(pn, extra) * 0.1])

            result = fast_als(U_init, V_init, W_init, max_iters=max_iters)
            if result is not None:
                return result

            # Also try pure random for lower ranks
            if rank < base:
                Ur = np.random.randn(nm, rank) * 0.5
                Vr = np.random.randn(mp, rank) * 0.5
                Wr = np.random.randn(pn, rank) * 0.5
                result = fast_als(Ur, Vr, Wr, max_iters=max_iters)
                if result is not None:
                    return result

        return None

    # Try to go below the structured rank
    lower_bound = max(n * m, m * p, p * n)  # theoretical lower bound
    # But our structured rank is n*m = 12
    # The true lower bound for matrix mult is max(nm, mp, pn) in some cases, but often lower
    # Known: rank(M_{n,m,p}) >= nm + mp + pn - nm - mp - pn + 1 is not the bound
    # Actually lower bound is max(nm, mp, pn) only in very specific cases
    # For now, try ranks from some reasonable lower bound up to best_rank - 1
    trial_ranks = list(range(max(6, best_rank - 8), best_rank))
    # Also try some very low ranks
    for tr in range(4, max(6, best_rank - 8)):
        if tr not in trial_ranks:
            trial_ranks.append(tr)
    trial_ranks = sorted(set(trial_ranks))

    for target_rank in trial_ranks:
        if time.time() > deadline:
            break
        res = try_rank(target_rank, n_attempts=2, max_iters=60)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
            break

    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
