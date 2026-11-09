import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580.0  # safety margin before 630s kill

    # ---------- Base: all weight-2 vectors (220 points, known valid) ----------
    base_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_list.append(p)
    base_points = np.array(base_list, dtype=np.int64)
    best_points = base_points.copy()
    best_k = base_points.shape[0]  # 220

    # ---------- Candidate generation ----------
    def generate_candidates(weights):
        cands = []
        for w in weights:
            for combo in combinations(range(d), w):
                for mask in range(1 << w):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, pos in enumerate(combo):
                        v[pos] = 1 if (mask >> idx) & 1 else -1
                    cands.append(v)
        return np.array(cands, dtype=np.int64)

    weight_sets = [
        [2],           # 220  (base, sanity check)
        [1, 2],        # 242
        [2, 3],        # 1540
        [1, 2, 3],     # 1562
        [2, 3, 4],     # 6820
        [1, 2, 3, 4],  # 6842
    ]

    rng = random.Random(42)

    # ---------- Greedy with vectorized distance checks ----------
    def greedy_once(cands, norms, order):
        selected_idx = []
        for idx in order:
            if time.time() > deadline - 15:
                break
            p = cands[idx]
            pn = norms[idx]
            if not selected_idx:
                selected_idx.append(idx)
                continue
            sel = cands[selected_idx]          # (m, d)
            sel_norms = norms[selected_idx]    # (m,)
            # dist^2 = ||p||^2 + ||s||^2 - 2 p.s   (vectorized over all selected)
            dots = sel @ p
            dist_sq = pn + sel_norms - 2 * dots
            # Need every pair's dist^2 >= max(max selected norm, pn)
            required = max(int(sel_norms.max()), int(pn))
            if dist_sq.min() >= required:
                selected_idx.append(idx)
        return selected_idx

    # ---------- Local search: try to swap a selected point for a better one ----------
    def local_search(cands, norms, selected_idx, max_iters=40):
        sel = list(selected_idx)
        m = len(sel)
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            sel_arr = cands[sel]
            sel_norms = norms[sel]
            # For each candidate, find how many selected it conflicts with
            # conflict if dist^2 < max(selected norms, cand norm)
            # We want a candidate that conflicts with 0 and is not already selected.
            # Vectorize: for each cand c, compute min over sel of (dist^2 - required_c)
            # required_c = max(max sel norm, norm c)
            cand_norms = norms
            max_sel_norm = int(sel_norms.max())
            required = np.maximum(max_sel_norm, cand_norms)  # (n,)
            # dist_sq[i, j] = norm[i] + sel_norms[j] - 2 * cands[i] @ sel_arr[j]
            dots = cands @ sel_arr.T                       # (n, m)
            dist_sq = cand_norms[:, None] + sel_norms[None, :] - 2 * dots  # (n, m)
            # conflict count per candidate
            conflicts = (dist_sq < required[:, None]).sum(axis=1)
            in_sel = np.zeros(len(cands), dtype=bool)
            in_sel[sel] = True
            conflicts[in_sel] = 1  # exclude already selected
            # Candidates with 0 conflicts are "free additions"
            free = np.where(conflicts == 0)[0]
            if len(free) > 0:
                # Add the one with the smallest norm (or random) to keep max norm low
                # Actually adding any free point increases k; pick smallest norm to be safe
                free_norms = norms[free]
                pick = free[np.argmin(free_norms)]
                sel.append(int(pick))
                continue
            # Try swap: remove the point that causes most conflicts, add a free one
            # Find the selected point with the highest "conflict pressure"
            # For each selected j, count how many free candidates would conflict with it
            # Simpler: try removing each selected point and re-checking a random free candidate
            # Pick a random selected to remove
            rem = rng.choice(range(m))
            new_sel = sel[:rem] + sel[rem+1:]
            # After removal, recompute free candidates
            if not new_sel:
                sel = new_sel
                continue
            new_sel_arr = cands[new_sel]
            new_sel_norms = norms[new_sel]
            new_max_sel_norm = int(new_sel_norms.max())
            new_required = np.maximum(new_max_sel_norm, cand_norms)
            new_dots = cands @ new_sel_arr.T
            new_dist_sq = cand_norms[:, None] + new_sel_norms[None, :] - 2 * new_dots
            new_conflicts = (new_dist_sq < new_required[:, None]).sum(axis=1)
            in_new = np.zeros(len(cands), dtype=bool)
            in_new[new_sel] = True
            new_conflicts[in_new] = 1
            new_free = np.where(new_conflicts == 0)[0]
            if len(new_free) > 0:
                # Pick a free candidate with smallest norm
                nf = new_free[np.argmin(norms[new_free])]
                new_sel.append(int(nf))
                if len(new_sel) >= len(sel):
                    sel = new_sel
                    m = len(sel)
                else:
                    # revert
                    pass
            else:
                pass
        return sel

    # ---------- Main search loop ----------
    for ws in weight_sets:
        if time.time() > deadline - 30:
            break
        cands = generate_candidates(ws)
        n = cands.shape[0]
        norms = (cands * cands).sum(axis=1).astype(np.int64)

        # Multiple restarts with different orderings
        max_restarts = 20 if n <= 2000 else 8
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            if restart % 3 == 0:
                order = list(range(n))
                rng.shuffle(order)
            elif restart % 3 == 1:
                order = sorted(range(n), key=lambda i: (int(norms[i]), i))
            else:
                order = sorted(range(n), key=lambda i: (-int(norms[i]), i))

            selected_idx = greedy_once(cands, norms, order)
            if len(selected_idx) > best_k:
                # Try to improve with local search
                improved = local_search(cands, norms, selected_idx, max_iters=30)
                if len(improved) >= len(selected_idx):
                    selected_idx = improved
                best_k = len(selected_idx)
                best_points = cands[selected_idx].copy()

    # ---------- Final safety: ensure best_points is valid ----------
    if best_points is None or best_points.shape[0] < 2:
        best_points = base_points
        best_k = base_points.shape[0]

    return np.array(best_points, dtype=np.int64)
