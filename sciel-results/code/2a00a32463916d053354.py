import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 143.0   # safety margin under 150s hard kill
    D = 11

    # ---- Build supports and vectorized conflict graph (intersection >= 3) ----
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)

    # One-hot matrix M[i, j] = 1 iff j in supports[i]
    M = np.zeros((N_sup, D), dtype=np.int8)
    for i, s in enumerate(supports):
        for j in s:
            M[i, j] = 1
    # Intersection sizes
    inter = M @ M.T  # (N_sup, N_sup), entries in {0,1,2,3,4}
    conflict = (inter >= 3)
    # Precompute neighbor lists and conflict matrix as boolean
    neighbor = [np.flatnonzero(conflict[i]) for i in range(N_sup)]
    # conflict_mat: boolean matrix, conflict_mat[i,j] = True if i conflicts with j
    conflict_mat = conflict

    def greedy_min_degree(avail_set, rng):
        """Greedy independent set via minimum-degree heuristic (vectorized degrees)."""
        avail = np.array(sorted(avail_set), dtype=np.int64)
        selected = []
        # degree[u] = number of neighbors of u that are still available
        deg = np.zeros(N_sup, dtype=np.int64)
        # Initialize degrees for available nodes
        for u in avail:
            deg[u] = conflict_mat[u, avail].sum()
        
        while avail.size > 0:
            if time.time() > deadline - 12.0:
                break
            m = deg[avail].min()
            cands = avail[deg[avail] == m]
            u = int(cands[rng.integers(len(cands))])
            selected.append(u)
            # remove u and its neighbors from avail
            to_remove_mask = np.zeros(N_sup, dtype=bool)
            to_remove_mask[u] = True
            to_remove_mask[neighbor[u]] = True
            avail = avail[~to_remove_mask[avail]]
            # update degrees
            rem_is_neigh = conflict_mat[u, avail]
            if rem_is_neigh.any():
                deg[avail[rem_is_neigh]] -= 1
        return selected

    def local_search(selected_set, rng):
        """1-2 swaps: try removing one selected node and adding two compatible nodes."""
        selected = set(selected_set)
        if not selected:
            return selected
        improved = True
        stall = 0
        sel_arr = np.array(list(selected), dtype=np.int64)
        
        while improved and time.time() < deadline - 12.0:
            improved = False
            stall += 1
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for v in sel_list:
                if time.time() > deadline - 12.0:
                    break
                sel_minus_v = selected - {v}
                if not sel_minus_v:
                    # If removing v leaves empty set, any two non-conflicting nodes work
                    cands = [x for x in range(N_sup) if x != v and not conflict_mat[x, v]]
                    if len(cands) >= 2:
                        # Find two non-conflicting candidates
                        for i in range(len(cands)):
                            for j in range(i+1, len(cands)):
                                if not conflict_mat[cands[i], cands[j]]:
                                    selected = {cands[i], cands[j]}
                                    improved = True
                                    break
                            if improved:
                                break
                    continue
                
                # Vectorized: find nodes compatible with all of sel_minus_v
                smv_arr = np.array(list(sel_minus_v), dtype=np.int64)
                # A node x is compatible with sel_minus_v iff no conflict with any member
                # conflict_mat[x, smv_arr] should be all False
                # For all x in range(N_sup): check conflict_mat[x, smv_arr].any()
                # This is: for each x, does any column in conflict_mat[x, smv_arr] have True?
                # Equivalent: (conflict_mat[:, smv_arr] & ~np.isin(np.arange(N_sup), np.concatenate([sel_minus_v, [v]]))).all(axis=1)
                
                # More efficient: compute for all x whether x conflicts with any in smv_arr
                # conflict_with_smv = conflict_mat[:, smv_arr].any(axis=1)
                # Then candidates are those not in selected, not v, and not conflicting with smv
                conflict_with_smv = conflict_mat[:, smv_arr].any(axis=1)
                in_selected_or_v = np.zeros(N_sup, dtype=bool)
                in_selected_or_v[list(selected)] = True
                in_selected_or_v[v] = True
                
                cands_mask = ~conflict_with_smv & ~in_selected_or_v
                cands = np.flatnonzero(cands_mask)
                
                if len(cands) < 2:
                    continue
                
                # Find two non-conflicting candidates
                found = False
                n = len(cands)
                # Vectorized pair check: conflict_mat[cands[:,None], cands[None,:]]
                # But this could be large. Use a smarter approach.
                for i in range(n):
                    if time.time() > deadline - 12.0:
                        break
                    u1 = cands[i]
                    # Find cands[j] for j > i that don't conflict with u1
                    non_conflict_with_u1 = ~conflict_mat[u1, cands[i+1:]]
                    if non_conflict_with_u1.any():
                        u2 = cands[i+1 + np.flatnonzero(non_conflict_with_u1)[0]]
                        selected.discard(v)
                        selected.add(u1)
                        selected.add(u2)
                        improved = True
                        found = True
                        break
            if not improved and stall > 5:
                break
        return selected

    def polish(selected_set):
        """Add any non-conflicting node that can be added."""
        selected = set(selected_set)
        if not selected:
            return selected
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            sel_arr = np.array(list(selected), dtype=np.int64)
            conflict_with_sel = conflict_mat[:, sel_arr].any(axis=1)
            in_sel = np.zeros(N_sup, dtype=bool)
            in_sel[list(selected)] = True
            addable_mask = ~conflict_with_sel & ~in_sel
            addable = np.flatnonzero(addable_mask)
            if addable.size > 0:
                # Add all addable nodes at once (they're all compatible with current set)
                # But need to check they're compatible with each other too
                # Actually, if they're all compatible with the current set, we need to check
                # pairwise compatibility among addable nodes
                for x in addable:
                    # Check x is compatible with all other addable nodes not yet added
                    # Simplify: add one at a time, recompute
                    selected.add(int(x))
                    changed = True
                    # Recompute for next iteration
                    break
        return selected

    best_iset = set()
    best_size = 0
    rng = np.random.default_rng(42)

    n_restarts = 400
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        avail = set(range(N_sup))
        iset = greedy_min_degree(avail, rng)
        iset_set = set(iset)
        iset_set = local_search(iset_set, rng)
        if len(iset_set) > best_size:
            best_size = len(iset_set)
            best_iset = set(iset_set)
            if time.time() < deadline - 20.0:
                iset2 = local_search(iset_set, rng)
                if len(iset2) > best_size:
                    best_size = len(iset2)
                    best_iset = set(iset2)
        if r % 25 == 24 and best_iset and time.time() < deadline - 15.0:
            imp = local_search(best_iset, rng)
            if len(imp) > best_size:
                best_size = len(imp)
                best_iset = set(imp)

    # Final exact polish pass on the best set (add any non-conflicting node)
    if time.time() < deadline - 5.0 and best_iset:
        best_iset = polish(best_iset)

    # ---- Assemble vectors ----
    vectors = []
    # Type A: ±2 e_i  (22 vectors), always compatible
    for i in range(D):
        for sgn in (1, -1):
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * sgn
            vectors.append(v)
    # Type D: 16 sign vectors per selected support
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
