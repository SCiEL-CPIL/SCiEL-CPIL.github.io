import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 585.0  # safety margin before 630s kill

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
        [2],           # 220  (base)
        [1, 2],        # 242
        [2, 3],        # 1540
        [1, 2, 3],     # 1562
        [2, 3, 4],     # 6820
        [1, 2, 3, 4],  # 6842
    ]

    rng = random.Random(42)

    # ---------- Vectorized greedy selection ----------
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
            sel = cands[selected_idx]
            sel_norms = norms[selected_idx]
            dots = sel @ p
            dist_sq = pn + sel_norms - 2 * dots
            required = max(int(sel_norms.max()), int(pn))
            if dist_sq.min() >= required:
                selected_idx.append(idx)
        return selected_idx

    # ---------- Local search: try to add or swap points ----------
    def local_search(cands, norms, selected_idx, max_iters=50):
        sel = list(selected_idx)
        m = len(sel)
        n = len(cands)
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            sel_arr = cands[sel]
            sel_norms = norms[sel]
            max_sel_norm = int(sel_norms.max())
            
            # Compute conflicts for all candidates
            dots = cands @ sel_arr.T
            dist_sq = norms[:, None] + sel_norms[None, :] - 2 * dots
            required = np.maximum(max_sel_norm, norms)
            conflicts = (dist_sq < required[:, None]).sum(axis=1)
            
            in_sel = np.zeros(n, dtype=bool)
            in_sel[sel] = True
            conflicts[in_sel] = 1
            
            free = np.where(conflicts == 0)[0]
            
            if len(free) > 0:
                # Add a free point with smallest norm
                free_norms = norms[free]
                pick = free[np.argmin(free_norms)]
                sel.append(int(pick))
                m += 1
                continue
            
            # Try swap: remove a selected point, see if we can add more
            # Find selected point that causes most conflicts
            # For each selected j, count how many candidates conflict only due to j
            # Simplified: try removing each selected point and see if free candidates appear
            for rem_idx in range(m):
                if time.time() > deadline - 25:
                    break
                new_sel = sel[:rem_idx] + sel[rem_idx+1:]
                if not new_sel:
                    continue
                new_sel_arr = cands[new_sel]
                new_sel_norms = norms[new_sel]
                new_max_sel_norm = int(new_sel_norms.max())
                new_dots = cands @ new_sel_arr.T
                new_dist_sq = norms[:, None] + new_sel_norms[None, :] - 2 * new_dots
                new_required = np.maximum(new_max_sel_norm, norms)
                new_conflicts = (new_dist_sq < new_required[:, None]).sum(axis=1)
                new_in_sel = np.zeros(n, dtype=bool)
                new_in_sel[new_sel] = True
                new_conflicts[new_in_sel] = 1
                new_free = np.where(new_conflicts == 0)[0]
                
                if len(new_free) > 0:
                    # Add up to 2 free points to compensate for removal
                    added = 0
                    nf = list(new_free)
                    rng.shuffle(nf)
                    for cand_idx in nf:
                        if added >= 2:
                            break
                        new_sel.append(int(cand_idx))
                        added += 1
                    if len(new_sel) >= len(sel):
                        sel = new_sel
                        m = len(sel)
                        break
        return sel

    # ---------- Main search loop ----------
    for ws in weight_sets:
        if time.time() > deadline - 30:
            break
        cands = generate_candidates(ws)
        n = cands.shape[0]
        norms = (cands * cands).sum(axis=1).astype(np.int64)

        max_restarts = 25 if n <= 2000 else 10
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
            
            # Try to improve with local search
            improved = local_search(cands, norms, selected_idx, max_iters=40)
            if len(improved) > len(selected_idx):
                selected_idx = improved
            
            if len(selected_idx) > best_k:
                best_k = len(selected_idx)
                best_points = cands[selected_idx].copy()

    # ---------- Final safety ----------
    if best_points is None or best_points.shape[0] < 2:
        best_points = base_points
        best_k = base_points.shape[0]

    return np.array(best_points, dtype=np.int64)
