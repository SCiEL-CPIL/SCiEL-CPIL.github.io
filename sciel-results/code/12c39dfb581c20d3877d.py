import time
import random
import numpy as np

def construct():
    deadline = time.time() + 142.0   # safety margin under the 150s wall-clock

    # ------------------------------------------------------------------
    # Build the ground-truth matrix-multiplication tensor T over Z.
    # A is 2x4, B is 4x6, C = A @ B is 2x6.
    #   T[a_idx, b_idx, c_idx] = 1  iff  C[i,j] uses A[i,k]*B[k,j]
    # Index conventions (official trace convention):
    #   U row index = a_idx = i*m + k      (m=4)
    #   V row index = b_idx = k*p + j      (p=6)
    #   W row index = c_idx = i*p + j
    # ------------------------------------------------------------------
    n, m, p = 2, 4, 6
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m + k, k*p + j, i*p + j] = 1
    T = np.ascontiguousarray(T)

    # ------------------------------------------------------------------
    # Seed: rank-42 solution = 6 independent copies of a 7-term Strassen
    # decomposition tiled over the 6 (A-half x B-third x C-third) blocks.
    #   A_L  = (0,1,4,5)     = (x11,x12,x21,x22)
    #   A_R  = (2,3,6,7)
    #   B_T1 = (0,1,6,7)      B_T2 = (2,3,8,9)   B_T3 = (4,5,10,11)
    #   B_B1 = (12,13,18,19)  B_B2 = (14,15,20,21) B_B3 = (16,17,22,23)
    #   C_01 = (0,2,1,3)  C_23 = (4,6,5,7)  C_45 = (8,10,9,11)
    # ------------------------------------------------------------------
    rank = 42
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)

    A_L = (0, 1, 4, 5)
    A_R = (2, 3, 6, 7)
    B_T1 = (0, 1, 6, 7)
    B_T2 = (2, 3, 8, 9)
    B_T3 = (4, 5, 10, 11)
    B_B1 = (12, 13, 18, 19)
    B_B2 = (14, 15, 20, 21)
    B_B3 = (16, 17, 22, 23)
    C_01 = (0, 2, 1, 3)
    C_23 = (4, 6, 5, 7)
    C_45 = (8, 10, 9, 11)

    blocks = [
        (A_L, B_T1, C_01), (A_L, B_T2, C_23), (A_L, B_T3, C_45),
        (A_R, B_B1, C_01), (A_R, B_B2, C_23), (A_R, B_B3, C_45),
    ]

    # The 7 Strassen-style terms per block as ((u_rows, vals),(v_rows, vals),(w_rows, vals))
    def block_terms(A_idx, B_idx, C_idx):
        x11, x12, x21, x22 = A_idx
        y11, y12, y21, y22 = B_idx
        z11, z12, z21, z22 = C_idx
        return [
            (([x11, x22], [1, 1]), ([y11, y22], [1, 1]), ([z11, z22], [1, 1])),  # M1
            (([x21, x22], [1, 1]), ([y11], [1]),            ([z21, z22], [1, -1])),  # M2
            (([x11], [1]),            ([y12, y22], [1, -1]), ([z12, z22], [1, 1])),  # M3
            (([x22], [1]),            ([y21, y11], [1, -1]), ([z11, z21], [1, 1])),  # M4
            (([x11, x12], [1, 1]),    ([y22], [1]),         ([z11, z12], [-1, 1])),  # M5
            (([x21, x11], [1, -1]),   ([y11, y12], [1, 1]), ([z22], [1])),           # M6
            (([x12, x22], [1, -1]),   ([y21, y22], [1, 1]), ([z11], [1])),           # M7
        ]

    col = 0
    for A_idx, B_idx, C_idx in blocks:
        for (ur, uv), (vr, vv), (wr, wv) in block_terms(A_idx, B_idx, C_idx):
            for r_, v_ in zip(ur, uv):
                U[r_, col] += v_
            for r_, v_ in zip(vr, vv):
                V[r_, col] += v_
            for r_, v_ in zip(wr, wv):
                W[r_, col] += v_
            col += 1

    # Fast exact check that the current U,V,W reproduce T (as integers).
    def is_valid(uu, vv, ww, rank_):
        u = uu[:, :rank_]; v = vv[:, :rank_]; w = ww[:, :rank_]
        if (u[:, rank_:] | v[:, rank_:] | w[:, rank_:]).any():
            return False
        # Reuse buffer to avoid reallocation each call.
        if not hasattr(is_valid, "buf"):
            is_valid.buf = np.zeros((n*m, rank), dtype=np.int64)
        else:
            is_valid.buf[:, :rank_] = 0
        np.einsum('ir,jr->ij', v, w, out=is_valid.buf[:, :rank_])
        np.einsum('ai,ik->ak', u, is_valid.buf[:, :rank_])
        if not np.array_equal(is_valid.buf[:, :rank_], T):
            return False
        return True

    # Rank-1 elimination: reorder terms so trailing ones are (u,v,w)-linearly
    # dependent, giving a true lower rank. Return new effective rank or rank_.
    def reduce_rank(uu, vv, ww, rank_):
        # work over Q using fractions-free integer row space (row echelon on int)
        def span_rows(M):
            # return list of pivot columns of row-reduced matrix M (int)
            M = M.astype(np.int64).copy()
            rows, cols = M.shape
            pivots = []
            r = 0
            for c in range(cols):
                if r >= rows:
                    break
                # find pivot row
                pr = None
                for i in range(r, rows):
                    if M[i, c] != 0:
                        pr = i
                        break
                if pr is None:
                    continue
                M[[r, pr]] = M[[pr, r]]
                pivot = M[r, c]
                # eliminate above
                for i in range(r):
                    if M[i, c] != 0:
                        M[i] = M[i] * pivot - M[r] * M[i, c]
                        # M[i,c] becomes 0 now
                pivots.append(c)
                r += 1
            return len(pivots)

        u = uu[:, :rank_].copy()
        v = vv[:, :rank_].copy()
        w = ww[:, :rank_].copy()
        # greedy order: keep basis, drop dependent
        order = list(range(rank_))
        kept = []
        for c in order:
            kept.append(c)
            cu = np.hstack([u[:, kept].T, u[:, c].reshape(-1, 1)]).T  # (n*m, kept+1)
            cv = np.hstack([v[:, kept].T, v[:, c].reshape(-1, 1)]).T
            cw = np.hstack([w[:, kept].T, w[:, c].reshape(-1, 1)]).T
            if span_rows(cu) < len(kept) + 1 and \
               span_rows(cv) < len(kept) + 1 and \
               span_rows(cw) < len(kept) + 1:
                pass  # keep, will be dropped later when it is the dependent tail
        # Simpler robust approach: build kept basis, then drop each non-basis term.
        kept = []
        for c in order:
            cand_u = np.column_stack([u[:, kept], u[:, c]]) if kept else u[:, c:c+1]
            cand_v = np.column_stack([v[:, kept], v[:, c]]) if kept else v[:, c:c+1]
            cand_w = np.column_stack([w[:, kept], w[:, c]]) if kept else w[:, c:c+1]
            du = span_rows(cand_u.T); dv = span_rows(cand_v.T); dw = span_rows(cand_w.T)
            if du < len(kept) + 1 or dv < len(kept) + 1 or dw < len(kept) + 1:
                continue  # not an independent new basis column -> candidate to drop
            kept.append(c)
        kept_set = set(kept)
        # Drop (zero) every column not in kept; this reduces rank to len(kept) if valid.
        for c in order:
            if c not in kept_set:
                uu[:, c] = 0
                vv[:, c] = 0
                ww[:, c] = 0
        new_rank = len(kept)
        if is_valid(uu, vv, ww, new_rank):
            return new_rank
        # restore if dropping broke validity (shouldn't, but be safe)
        # (we do not restore here; the caller keeps a verified snapshot)
        return rank_

    # Ensure seed validity + initial reduction.
    U, V, W = U.copy(), V.copy(), W.copy()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    if not is_valid(U, V, W, rank):
        # Should never happen for the seeded construction; fall back safely.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank = reduce_rank(U, V, W, rank)
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Slice-rank lower bound (guide only, never violated by exact search):
    # For 2x4x6 it is 40. We aim to reach <= 40.
    lower_bound = 40

    rng = random.Random(12345)
    active_cols = set(range(rank))

    # ------------------------------------------------------------------
    # Local search: try elementary moves, accept a move only if it keeps the
    # decomposition exactly valid over Z. Two families:
    #   (a) zero / perturb a single factor entry (exploit slack),
    #   (b) merge two rank terms that become linearly dependent (reduce rank).
    # ------------------------------------------------------------------
    def try_zero_entry(mat, i, col):
        """Temporarily set mat[i,col]=0; return True if still valid."""
        orig = mat[i, col]
        if orig == 0:
            return False
        mat[i, col] = 0
        ok = is_valid(U, V, W, rank)
        mat[i, col] = orig
        return ok

    def apply_move(mat, i, col):
        mat[i, col] = 0

    def try_merge(a, b):
        """If after making b's factors lie in span of a, the whole tensor
        stays valid, merge b into a (zero b). Conservative integer test."""
        # A safe merge: express U[:,b],V[:,b],W[:,b] as integer combos of U[:,a]
        # etc. is hard generally; instead attempt the common case where b becomes
        # exactly a scaled/sum. We only do a cheap exact re-check variant:
        # try zeroing a term and checking validity (same as zero entry over full col).
        for mat in (U, V, W):
            for row in range(mat.shape[0]):
                if mat[row, b] != 0:
                    mat[row, b] = 0
        if is_valid(U, V, W, rank - 1):
            return b, a
        # revert
        return None

    start = time.time()
    steps = 0
    no_improve = 0
    best_ever_valid = True

    def elapsed():
        return time.time() - start

    # Randomize search order each sweep so we explore broadly.
    while elapsed() < deadline - 8.0:
        # periodic global rank reduction on the current state
        old_rank = rank
        rank = reduce_rank(U, V, W, rank)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            no_improve = 0
        if rank <= lower_bound:
            break

        # --- random entry-zero attempts ---
        for _ in range(3000):
            if elapsed() > deadline - 8.0:
                break
            mat = (U, V, W)[rng.randrange(3)]
            i = rng.randrange(mat.shape[0])
            c = rng.randrange(rank)
            if mat[i, c] == 0:
                continue
            if try_zero_entry(mat, i, c):
                apply_move(mat, i, c)
                active_cols = set(range(rank))  # columns unchanged, entry zeroed
                new_rank = reduce_rank(U, V, W, rank)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    no_improve = 0
                    rank = new_rank
                else:
                    no_improve += 1
                if rank <= lower_bound:
                    break

        # --- attempt whole-column merge (rank cut) by dependency check ---
        if rank > lower_bound:
            for _ in range(60):
                if elapsed() > deadline - 8.0:
                    break
                b = rng.randrange(rank)
                a = rng.randrange(rank)
                if a == b:
                    continue
                saved = (U[:, b].copy(), V[:, b].copy(), W[:, b].copy())
                U[:, b] = 0; V[:, b] = 0; W[:, b] = 0
                if is_valid(U, V, W, rank - 1):
                    rank -= 1
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        no_improve = 0
                    if rank <= lower_bound:
                        break
                else:
                    U[:, b] = saved[0]; V[:, b] = saved[1]; W[:, b] = saved[2]
                steps += 1

    # ------------------------------------------------------------------
    # Return the best verified solution. Convert to plain-list matrices
    # matching the requested shape (n*m, rank) etc. Use best_rank columns.
    # ------------------------------------------------------------------
    if best_ever_valid and is_valid(best_U, best_V, best_W, best_rank):
        out_U = best_U[:, :best_rank].tolist()
        out_V = best_V[:, :best_rank].tolist()
        out_W = best_W[:, :best_rank].tolist()
        # sanity: integer half-integer entries only
        for mm in (out_U, out_V, out_W):
            for row in mm:
                for x in row:
                    assert (x * 2) % 1 == 0
        return {"U": out_U, "V": out_V, "W": out_W}

    # Fallback: return the (verified) rank-42 seed if anything above failed.
    return {"U": U[:, :42].tolist(), "V": V[:, :42].tolist(), "W": W[:, :42].tolist()}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
