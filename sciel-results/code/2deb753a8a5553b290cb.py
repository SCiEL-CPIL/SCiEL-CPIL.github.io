import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 140.0


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = DEADLINE

    # Strassen 2x2x2 decomposition (7 terms)
    strassen_decomp = [
        ({0: 1, 3: 1},   {0: 1, 3: 1},   {0: 1, 3: 1}),
        ({2: 1, 3: 1},   {0: 1},         {2: 1, 3: -1}),
        ({0: 1},         {1: 1, 3: -1},  {1: 1, 3: 1}),
        ({3: 1},         {2: 1, 0: -1},  {0: 1, 2: 1}),
        ({0: 1, 1: 1},   {3: 1},         {1: 1, 0: -1}),
        ({2: 1, 0: -1},  {0: 1, 1: 1},   {3: 1}),
        ({1: 1, 3: -1},  {2: 1, 3: 1},   {0: 1}),
    ]

    def _build_block_strassen():
        """n=3 -> 2+1, m=4 -> 2+2, p=8 -> 4x2. Rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[li_i * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + li_i][col] += cf
                    col += 1
            for k in range(8):
                for j2 in range(2):
                    gj = 2 * j1 + j2
                    U[2 * m + gj][col] = 1
                    V[gj * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_alt_split():
        """n=3 -> 1+2. Top row 0 schoolbook (32); bottom rows 1,2 Strassen (56)."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += cf
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_4x4_blocks():
        """n=3 -> 1+2, m=4 -> 2+2, p=8 -> 2x4. Rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j1 in range(2):
            for l1 in range(2):
                for j2 in range(2):
                    for k2 in range(4):
                        gj = 2 * j1 + j2
                        gk = 4 * l1 + k2
                        U[0 * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + 0][col] = 1
                        col += 1
                for l2 in range(2):
                    for u_c, v_c, w_c in strassen_decomp:
                        for li, cf in u_c.items():
                            li_i, li_j = li // 2, li % 2
                            U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                        for li, cf in v_c.items():
                            li_i, li_j = li // 2, li % 2
                            V[(2 * j1 + li_i) * p + (4 * l1 + 2 * l2 + li_j)][col] += cf
                        for li, cf in w_c.items():
                            li_i, li_j = li // 2, li % 2
                            W[(4 * l1 + 2 * l2 + li_j) * n + (1 + li_i)][col] += cf
                        col += 1
        return {"U": U, "V": V, "W": W}

    def _build_symmetric_split():
        """Try a different partition: n=3 -> 2+1 but m=4 -> 4 (no m-split), p=8 -> 4x2.
        For the 2x4x8 block: use 2x2 Strassen on n-p subblocks with m as free dim.
        Actually, let's try: for each of 4 m-values, do 2x2x8 Strassen (4 blocks of 2x2x2).
        That gives 4*4*7 = 112 for top, plus 4*8 = 32 for bottom = 144. Too high.
        Instead: n=3->2+1, m=4->4, p=8->4x2.
        Top 2x4x8: 4 m-values, each 2x2x2 Strassen -> 4*7=28 per m-block? No.
        Let's just do schoolbook for top: 2*4*8=64, bottom: 1*4*8=32, total 96.
        This is just schoolbook, useful as fallback."""
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

    def _verify(U, V, W, tol=1e-9):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _rank_of(mat):
        return np.linalg.matrix_rank(np.array(mat, dtype=float), tol=1e-9)

    def _reduce(U, V, W, budget):
        """Vectorized multi-pass greedy column removal with tensor checks."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        keep = list(range(r))

        for _pass in range(10):
            if time.time() > budget:
                break
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            r = Ua.shape[1]
            if r == 0:
                break

            T_total = np.zeros((n, m, p))
            contributions = []
            for c in range(r):
                tc = Ua[:, c].reshape(n, m, 1) * Va[:, c].reshape(1, m, p) * Wa[:, c].reshape(p, 1, n)
                contributions.append(tc)
                T_total += tc

            removed_any = False
            for idx in range(r):
                if time.time() > budget:
                    break
                t_remaining = T_total - contributions[idx]
                if np.allclose(t_remaining, 1.0, atol=1e-9):
                    keep = [c for c in keep if c != idx]
                    removed_any = True
                    break

            if removed_any:
                continue

            # Try removing pairs
            found_pair = False
            for i in range(r):
                if time.time() > budget:
                    break
                for j in range(i + 1, r):
                    if np.allclose(contributions[i] + contributions[j], 0.0, atol=1e-9):
                        keep = [c for c in keep if c != i and c != j]
                        found_pair = True
                        break
                if found_pair:
                    break
            if found_pair:
                continue

            # Linear dependency check: if column c's U vector is in span of others,
            # and same for V and W, then it might be removable (but tensor check is needed)
            # For now, break if nothing else found
            break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    def _reduce_ld(U, V, W, budget):
        """Linear dependency based reduction: remove columns whose factor vectors
        are linearly dependent with others, then verify tensor identity still holds."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        keep = list(range(r))

        for _pass in range(8):
            if time.time() > budget:
                break
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            r = Ua.shape[1]
            if r == 0:
                break

            # Find columns where U vector is in span of others
            removable_candidates = []
            for idx in range(r):
                if time.time() > budget:
                    break
                others = [c for c in range(r) if c != idx]
                if not others:
                    continue
                # Check if U[:,idx] is in span of U[:,others]
                U_others = Ua[:, others]
                U_idx = Ua[:, idx]
                # Use least squares to check
                if U_others.shape[1] == 0:
                    continue
                try:
                    sol, residuals, rank_u, sv = np.linalg.lstsq(U_others, U_idx, rcond=1e-10)
                    U_recon = U_others @ sol
                    if np.allclose(U_recon, U_idx, atol=1e-9):
                        # Also check V and W
                        V_others = Va[:, others]
                        V_idx = Va[:, idx]
                        sol_v, _, rank_v, _ = np.linalg.lstsq(V_others, V_idx, rcond=1e-10)
                        V_recon = V_others @ sol_v
                        if np.allclose(V_recon, V_idx, atol=1e-9):
                            W_others = Wa[:, others]
                            W_idx = Wa[:, idx]
                            sol_w, _, rank_w, _ = np.linalg.lstsq(W_others, W_idx, rcond=1e-10)
                            W_recon = W_others @ sol_w
                            if np.allclose(W_recon, W_idx, atol=1e-9):
                                removable_candidates.append(idx)
                except Exception:
                    continue

            # Try removing each candidate (tensor check)
            if removable_candidates:
                T_total = np.zeros((n, m, p))
                contributions = []
                for c in range(r):
                    tc = Ua[:, c].reshape(n, m, 1) * Va[:, c].reshape(1, m, p) * Wa[:, c].reshape(p, 1, n)
                    contributions.append(tc)
                    T_total += tc

                removed = False
                for idx in removable_candidates:
                    if time.time() > budget:
                        break
                    t_remaining = T_total - contributions[idx]
                    if np.allclose(t_remaining, 1.0, atol=1e-9):
                        keep = [c for c in keep if c != idx]
                        removed = True
                        break

                if removed:
                    continue

            break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    def _reduce_triples(U, V, W, budget):
        """Try removing triples of columns whose combined tensor contribution is zero."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        keep = list(range(r))

        for _pass in range(3):
            if time.time() > budget:
                break
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            r = Ua.shape[1]
            if r < 3:
                break

            T_total = np.zeros((n, m, p))
            contributions = []
            for c in range(r):
                tc = Ua[:, c].reshape(n, m, 1) * Va[:, c].reshape(1, m, p) * Wa[:, c].reshape(p, 1, n)
                contributions.append(tc)
                T_total += tc

            found = False
            for i in range(r):
                if time.time() > budget:
                    break
                for j in range(i + 1, r):
                    if time.time() > budget:
                        break
                    for k in range(j + 1, r):
                        if time.time() > budget:
                            break
                        if np.allclose(contributions[i] + contributions[j] + contributions[k], 0.0, atol=1e-9):
                            keep = [c for c in keep if c != i and c != j and c != k]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    best = None
    best_rank = 10 ** 9

    builders = [_build_block_strassen, _build_alt_split, _build_4x4_blocks, _build_symmetric_split]
    for builder in builders:
        if time.time() > deadline:
            break
        sol = builder()
        if sol is None:
            continue
        U, V, W = sol["U"], sol["V"], sol["W"]
        if not _verify(U, V, W):
            continue

        # Apply multiple reduction strategies in sequence
        cur_U, cur_V, cur_W, cur_rank = U, V, W, len(U[0])
        for reducer in (_reduce, _reduce_ld, _reduce_triples, _reduce):
            if time.time() > deadline:
                break
            try:
                U2, V2, W2, rk = reducer(cur_U, cur_V, cur_W, deadline)
                if rk < cur_rank and _verify(U2, V2, W2):
                    cur_U, cur_V, cur_W, cur_rank = U2, V2, W2, rk
            except Exception:
                pass

        if cur_rank < best_rank and _verify(cur_U, cur_V, cur_W):
            best = {"U": cur_U, "V": cur_V, "W": cur_W}
            best_rank = cur_rank

    if best is None:
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best
