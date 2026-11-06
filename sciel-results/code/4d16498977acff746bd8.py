"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 320.0  # safety margin before 330s wall-clock kill


def _iso_flags():
    """Return (iso, pair_iso):
       iso[i,j,k]  = True iff points i,j,k form an isosceles triangle.
       pair_iso[i,j] = bool array over k: iso[i,j,k].
    """
    pts = np.array([[x, y] for y in range(N) for x in range(N)], dtype=np.int64)
    n = N * N
    d = (pts[:, None, :] - pts[None, :, :]) ** 2
    d = d.sum(axis=2)  # n x n squared distances
    iso = (d[:, None, :] == d[None, :, :]) | (d[:, None, :] == d[:, None, :].T) \
        | (d[None, :, :] == d[:, None, :].T)
    iso = iso & ~((d == 0)[:, None, :]) & ~((d == 0)[None, :, :]) & ~((d == 0)[:, None, :].T)
    # pair_iso[i,j][k] = iso[i,j,k]
    pair_iso = iso.transpose(0, 2, 1).reshape(n, n, n)  # [i, j, k]
    return iso, pair_iso


def construct():
    t0 = time.time()
    iso, pair_iso = _iso_flags()
    n = N * N
    all_pts = np.arange(n)
    all_pts_list = list(all_pts)

    def blocked_mask_for_set(S):
        """Boolean array over all points: which are blocked by pairs in S."""
        b = np.zeros(n, dtype=bool)
        L = len(S)
        for ai in range(L):
            for bi in range(ai + 1, L):
                b |= pair_iso[S[ai], S[bi]]
        return b

    # ---- Greedy maximal construction: pick points that block most uncovered ----
    chosen = []
    chosen_set = set()
    covered = np.zeros(n, dtype=bool)
    for _ in range(n):
        if time.time() > DEADLINE - 20:
            break
        best = -1
        best_gain = -1
        for p in all_pts_list:
            if p in chosen_set:
                continue
            # points newly blocked by adding p = pairs(p, c) for c in chosen
            newblk = np.zeros(n, dtype=bool)
            for c in chosen:
                newblk |= pair_iso[p, c]
            gain = (~covered & newblk).sum()
            if gain > best_gain:
                best_gain = gain
                best = p
        if best < 0 or best_gain <= 0:
            break
        # add best
        chosen.append(best)
        chosen_set.add(best)
        for c in chosen[:-1]:
            covered |= pair_iso[best, c]

    # ---- Removal-based local search: shrink while preserving maximality ----
    chosen = list(chosen)
    improved = True
    while improved:
        if time.time() > DEADLINE - 5:
            break
        improved = False
        L = len(chosen)
        for idx in range(L):
            if time.time() > DEADLINE - 5:
                break
            cand = chosen[idx]
            rest = [c for j, c in enumerate(chosen) if j != idx]
            # points that ONLY cand blocks (with the rest)
            # covered_by_rest
            cov_rest = np.zeros(n, dtype=bool)
            for ai in range(len(rest)):
                for bi in range(ai + 1, len(rest)):
                    cov_rest |= pair_iso[rest[ai], rest[bi]]
            # what cand blocks with rest
            cand_blocks = np.zeros(n, dtype=bool)
            for c in rest:
                cand_blocks |= pair_iso[cand, c]
            # points blocked by cand but not by rest
            only = cand_blocks & ~cov_rest
            # maximality holds after removal iff no uncovered point can be added.
            # A point p (not in rest) can be added iff it is not blocked by rest,
            # i.e., p not in cov_rest and p not in rest.
            # If only is empty, removing cand keeps all previously-blocked points
            # blocked by rest -> maximality preserved (and no new point addable).
            if only.sum() == 0:
                # removing cand is safe
                del chosen[idx]
                improved = True
                break

    # ---- Final validation / ensure maximality (re-add if needed) ----
    if time.time() < DEADLINE - 2:
        chosen_set = set(chosen)
        cov = np.zeros(n, dtype=bool)
        for ai in range(len(chosen)):
            for bi in range(ai + 1, len(chosen)):
                cov |= pair_iso[chosen[ai], chosen[bi]]
        for p in all_pts_list:
            if p in chosen_set:
                continue
            # can p be added without creating isosceles with a pair in chosen?
            ok = True
            for ai in range(len(chosen)):
                for bi in range(ai + 1, len(chosen)):
                    if iso[p, chosen[ai], chosen[bi]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)

    # ---- Convert to (x,y) tuples ----
    pts_arr = np.array([[x, y] for y in range(N) for x in range(N)])
    result = [tuple(pts_arr[i]) for i in chosen]
    return result
# EVOLVE-BLOCK-END
