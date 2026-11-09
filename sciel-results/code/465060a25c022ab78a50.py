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

    rng = random.Random(42)

    # ---------- Helper: Check if a point can be added to a set ----------
    def can_add(p, pn, points, norms):
        if len(points) == 0:
            return True
        # max norm in points
        max_norm = int(norms.max())
        required = max(max_norm, int(pn))
        # dist^2 = pn + norms - 2 * dots
        dots = points @ p
        dist_sq = pn + norms - 2 * dots
        return dist_sq.min() >= required

    # ---------- Greedy addition to an existing set ----------
    def greedy_add_to_set(cands, norms, order, current_points, current_norms):
        # current_points: (m, d), current_norms: (m,)
        added_indices = []
        for idx in order:
            if time.time() > deadline - 15:
                break
            p = cands[idx]
            pn = norms[idx]
            if can_add(p, pn, current_points, current_norms):
                added_indices.append(idx)
                # Update current set
                current_points = np.vstack([current_points, p])
                current_norms = np.append(current_norms, pn)
        return added_indices, current_points, current_norms

    # ---------- Local Search ----------
    def local_search(cands, norms, selected_idx, max_iters=40):
        sel = list(selected_idx)
        m = len(sel)
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            sel_arr = cands[sel]
            sel_norms = norms[sel]
            
            # Find free candidates
            cand_norms = norms
            max_sel_norm = int(sel_norms.max())
            required = np.maximum(max_sel_norm, cand_norms)
            dots = cands @ sel_arr.T
            dist_sq = cand_norms[:, None] + sel_norms[None, :] - 2 * dots
            conflicts = (dist_sq < required[:, None]).sum(axis=1)
            in_sel = np.zeros(len(cands), dtype=bool)
            in_sel[sel] = True
            conflicts[in_sel] = 1
            free = np.where(conflicts == 0)[0]
            
            if len(free) > 0:
                pick = free[np.argmin(norms[free])]
                sel.append(int(pick))
                continue
            
            # Swap
            if m > 0:
                rem = rng.choice(range(m))
                new_sel = sel[:rem] + sel[rem+1:]
                if new_sel:
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
                        nf = new_free[np.argmin(norms[new_free])]
                        new_sel.append(int(nf))
                        if len(new_sel) >= len(sel):
                            sel = new_sel
                            m = len(sel)
        return sel

    # ---------- Main Search ----------
    # Strategy: Start with base, then add weight 3, 4, 5 candidates.
    
    # 1. Add weight 3 candidates
    cands_w3 = generate_candidates([3])
    norms_w3 = (cands_w3 * cands_w3).sum(axis=1).astype(np.int64)
    order_w3 = list(range(len(cands_w3)))
    rng.shuffle(order_w3)
    
    added_w3, new_points, new_norms = greedy_add_to_set(cands_w3, norms_w3, order_w3, best_points, (best_points*best_points).sum(axis=1).astype(np.int64))
    if len(new_points) > best_k:
        best_points = new_points
        best_k = len(new_points)
        # Local search
        # We need to map new_points back to a global candidate set for local search?
        # Or just run local search on the combined set?
        # For simplicity, we can just keep the points.
        # But local_search expects cands and norms arrays.
        # We can create a combined array of all candidates (base + w3 + w4) and indices.
        
    # To make local search effective, we should work with a single large candidate pool.
    # Let's generate a pool of weights 2, 3, 4.
    pool_cands = generate_candidates([2, 3, 4])
    pool_norms = (pool_cands * pool_cands).sum(axis=1).astype(np.int64)
    
    # Initialize selected with base points (indices 0 to 219 in pool, assuming order is preserved)
    # generate_candidates iterates weights then combos then masks.
    # Weight 2 comes first. So indices 0..219 are weight 2.
    base_indices = list(range(220))
    
    # Greedy add from pool
    order_pool = list(range(len(pool_cands)))
    rng.shuffle(order_pool)
    
    selected_idx, sel_points, sel_norms = greedy_add_to_set(pool_cands, pool_norms, order_pool, base_points, (base_points*base_points).sum(axis=1).astype(np.int64))
    
    # Note: greedy_add_to_set returns indices into cands, but we started with base_points.
    # The indices returned are into pool_cands.
    # But base_points are NOT in pool_cands at the same indices?
    # Yes, they are. pool_cands[0:220] are the weight 2 vectors.
    # So we need to combine base_indices and selected_idx.
    all_selected = base_indices + selected_idx
    
    # Check for duplicates? base_indices are 0..219. selected_idx are from 220 onwards (if greedy only added new ones).
    # Wait, greedy_add_to_set checks compatibility with current_points.
    # If a weight 2 point is already in current_points, it will be rejected.
    # So selected_idx will only contain indices > 219.
    
    if len(all_selected) > best_k:
        best_k = len(all_selected)
        best_points = pool_cands[all_selected]
        
        # Local search
        improved = local_search(pool_cands, pool_norms, all_selected, max_iters=50)
        if len(improved) > best_k:
            best_k = len(improved)
            best_points = pool_cands[improved]

    # Try weight 5?
    if time.time() < deadline - 60:
        pool_cands_5 = generate_candidates([2, 3, 4, 5])
        pool_norms_5 = (pool_cands_5 * pool_cands_5).sum(axis=1).astype(np.int64)
        # Re-initialize with current best?
        # This is complex. Stick to 2,3,4 for now.

    return np.array(best_points, dtype=np.int64)
