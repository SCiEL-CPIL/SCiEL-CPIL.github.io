"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, rank):
        """Vectorized exactness check of the decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_strassen_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_pat, v_pat, w_pat) mapping local coords to values."""
        terms = []
        u1 = {(0, 0): 1, (1, 1): 1}; v1 = {(0, 0): 1, (1, 1): 1}; w1 = {(0, 0): 1, (1, 1): 1}
        terms.append((u1, v1, w1))
        u2 = {(1, 0): 1, (1, 1): 1}; v2 = {(0, 0): 1}; w2 = {(0, 1): 1, (1, 1): -1}
        terms.append((u2, v2, w2))
        u3 = {(0, 0): 1}; v3 = {(0, 1): 1, (1, 1): -1}; w3 = {(1, 0): 1, (1, 1): 1}
        terms.append((u3, v3, w3))
        u4 = {(1, 1): 1}; v4 = {(1, 0): 1, (0, 0): -1}; w4 = {(0, 0): 1, (0, 1): 1}
        terms.append((u4, v4, w4))
        u5 = {(0, 0): 1, (0, 1): 1}; v5 = {(1, 1): 1}; w5 = {(0, 0): -1, (1, 0): 1}
        terms.append((u5, v5, w5))
        u6 = {(1, 0): 1, (0, 0): -1}; v6 = {(0, 0): 1, (0, 1): 1}; w6 = {(1, 1): 1}
        terms.append((u6, v6, w6))
        u7 = {(0, 1): 1, (1, 1): -1}; v7 = {(1, 0): 1, (1, 1): 1}; w7 = {(0, 0): 1}
        terms.append((u7, v7, w7))
        return terms

    def build_decomposition(i_groups, j_groups, k_groups, strassen_terms):
        """Build a block decomposition from given groupings.
        Returns (result_dict, rank) or (None, None)."""
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        U[ig[li] * m + jg[lj]][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        V[jg[lj] * p + kg[lk]][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        W[kg[lk] * n + ig[li]][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        if col != total_rank:
            return None, None
        return {"U": U, "V": V, "W": W}, total_rank

    def build_strassen_candidate(i_groups, j_groups, k_groups, strassen_terms):
        """Build the candidate term set (U,V,W as numpy) from the block partition.
        Returns (U_arr, V_arr, W_arr, rank)."""
        result, rank = build_decomposition(i_groups, j_groups, k_groups, strassen_terms)
        if result is None:
            return None
        return (np.array(result['U'], dtype=float),
                np.array(result['V'], dtype=float),
                np.array(result['W'], dtype=float), rank)

    def try_eliminate(U, V, W, rank, keep_mask, deadline):
        """Try to eliminate terms not in keep_mask using Gaussian elimination.
        Returns (U,V,W,new_rank) or None."""
        r0 = rank
        n_keep = int(keep_mask.sum())
        n_elim = r0 - n_keep
        if n_elim <= 0:
            return (U, V, W, r0)

        # Build the 90x90 constraint matrix T[c, (i,j,k)] = U[i,j,c]*V[j,k,c]*W[k,i,c]
        T = np.zeros((r0, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col_idx = i * m * p + j * p + k
                    T[:, col_idx] = U[i * m + j] * V[j * p + k] * W[k * n + i]

        keep_idx = np.where(keep_mask)[0]
        elim_idx = np.where(~keep_mask)[0]

        # For each eliminated term c, solve T[keep] * x = T[c]
        T_keep = T[keep_idx, :]
        # Precompute pseudoinverse of T_keep (90 constraints, n_keep unknowns)
        try:
            T_keep_pinv = np.linalg.pinv(T_keep, rcond=1e-12)
        except np.linalg.LinAlgError:
            return None

        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()

        for c in elim_idx:
            b = T[c, :]
            x = T_keep_pinv @ b
            # Check residual
            resid = np.linalg.norm(T_keep @ x - b)
            if resid > 1e-7:
                return None
            # Subtract x * term[keep] from term[c]
            for idx, kk in enumerate(keep_idx):
                new_U[:, c] -= x[idx] * new_U[:, kk]
                new_V[:, c] -= x[idx] * new_V[:, kk]
                new_W[:, c] -= x[idx] * new_W[:, kk]
            if time.time() > deadline:
                return None

        new_rank = r0
        return (new_U, new_V, new_W, new_rank)

    def finalize(U, V, W, rank):
        """Round near-integer entries and return dict."""
        U_r = np.round(U).astype(int).tolist()
        V_r = np.round(V).astype(int).tolist()
        W_r = np.round(W).astype(int).tolist()
        # Verify rounded version
        if verify_numpy(U_r, V_r, W_r, rank):
            return {"U": U_r, "V": V_r, "W": W_r}
        # Fall back to unrounded (should still be exact)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Main search ----
    best_result = schoolbook()
    best_rank = n * m * p

    strassen_terms = get_strassen_terms()

    # Primary partition
    i_groups = [[0, 1], [2]]
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4, 5]]

    cand = build_strassen_candidate(i_groups, j_groups, k_groups, strassen_terms)
    if cand is not None:
        U, V, W, rank = cand
        if verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank):
            # Try elimination: keep only the 54 terms for i in {0,1}
            # Terms are ordered by block: for each (ig, jg, kg) in row-major.
            # Block order:
            #   ig=[0,1]: jg=[0,1] kg=[0,1] (7), jg=[0,1] kg=[2,3] (7), jg=[0,1] kg=[4,5] (7),
            #             jg=[2,3] kg=[0,1] (7), jg=[2,3] kg=[2,3] (7), jg=[2,3] kg=[4,5] (7),
            #             jg=[4]   kg=[0,1] (4), jg=[4]   kg=[2,3] (4), jg=[4]   kg=[4,5] (4)  -> 54
            #   ig=[2]:   ... 30
            keep_mask = np.zeros(rank, dtype=bool)
            keep_mask[:54] = True  # first 54 terms are i in {0,1}
            elim = try_eliminate(U, V, W, rank, keep_mask, deadline)
            if elim is not None:
                new_U, new_V, new_W, new_rank = elim
                # Check exactness of the eliminated version
                if verify_numpy(new_U.tolist(), new_V.tolist(), new_W.tolist(), new_rank):
                    res = finalize(new_U, new_V, new_W, new_rank)
                    if new_rank < best_rank:
                        best_result = res
                        best_rank = new_rank

            # If elimination failed, keep the 84-term block result
            if best_rank > rank:
                res = finalize(U, V, W, rank)
                best_result = res
                best_rank = rank

    # Fallback: also try a few alternate partitions for robustness
    alternate_partitions = [
        ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
        ([[0, 1], [2]], [[0, 2], [1, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
        ([[0, 1], [2]], [[0, 1], [2, 3], [4]], [[0, 2], [1, 3], [4, 5]]),
    ]
    for (ig, jg, kg) in alternate_partitions:
        if time.time() > deadline:
            break
        cand = build_strassen_candidate(ig, jg, kg, strassen_terms)
        if cand is None:
            continue
        Ua, Va, Wa, ra = cand
        if verify_numpy(Ua.tolist(), Va.tolist(), Wa.tolist(), ra):
            if ra < best_rank:
                best_result = finalize(Ua, Va, Wa, ra)
                best_rank = ra

    return best_result
# EVOLVE-BLOCK-END
