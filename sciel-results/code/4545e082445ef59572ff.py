"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 115.0  # hard wall-clock safety margin before the 150 s limit

def construct():
    t0 = time.perf_counter()
    best = None
    best_k = 0

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    220 vectors, all mutually valid (min dist = sqrt(2) = max norm)
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    update_best(base_arr)

    # ------------------------------------------------------------------
    # 2) Generate candidate vectors across multiple shells
    # ------------------------------------------------------------------
    def gen_shell(m):
        """Generate all integer vectors in R^D with L2^2 = m."""
        results = []
        if m == 2:
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
        elif m == 3:
            for combo in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
        elif m == 4:
            for combo in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)
        return np.array(results, dtype=np.int64) if results else np.zeros((0, D), dtype=np.int64)

    # ------------------------------------------------------------------
    # 3) Core: minimum-degree greedy + swap local search on conflict graph
    # ------------------------------------------------------------------
    def solve_shell(V, max_ip_allowed):
        """Minimum-degree greedy then swap local search.
        V: (n, D) int array. max_ip_allowed: max inner product allowed between any pair.
        Returns best subset as (k, D) int array."""
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)

        # Compute Gram matrix
        G = V.astype(np.int64) @ V.astype(np.int64).T
        # Conflict: ip > max_ip_allowed
        conflict = G > max_ip_allowed
        # Self-conflict: set diagonal to False
        np.fill_diagonal(conflict, False)

        # Degree of each node = number of conflicts
        degree = conflict.sum(axis=1).astype(np.int32)

        # --- Minimum-degree greedy ---
        selected = set()
        remaining = set(range(n))
        # Use a list sorted by degree for efficiency; rebuild when needed
        # Track current degree for remaining nodes
        cur_degree = degree.copy()

        while remaining and not is_deadline():
            # Find node with minimum degree in remaining
            min_deg = min(cur_degree[i] for i in remaining)
            # Among those with min degree, pick the first (deterministic)
            candidates = [i for i in remaining if cur_degree[i] == min_deg]
            pick = min(candidates)  # deterministic tiebreak

            selected.add(pick)
            remaining.discard(pick)

            # Update degrees: for each remaining node, if it conflicts with pick, degree increases
            # Actually we track conflicts among remaining: when we pick, we remove it from remaining.
            # The "degree" in the greedy context is the number of conflicts with ALREADY SELECTED nodes.
            # Let me re-think: minimum-degree greedy for independent set means picking the node
            # with the fewest neighbors (conflicts) among remaining, then removing it and its neighbors.
            # But that's for finding a maximal independent set. We want a large independent set.
            
            # Actually, the standard minimum-degree heuristic for independent set:
            # Repeatedly pick the vertex of minimum degree in the remaining graph, add to independent set,
            # remove it and all its neighbors from the graph.
            
            # Remove all neighbors of pick from remaining
            neighbors = np.where(conflict[pick])[0]
            for nb in neighbors:
                if nb in remaining:
                    remaining.discard(nb)

        greedy_result = sorted(selected)
        if not greedy_result:
            return np.zeros((0, D), dtype=np.int64)

        # --- Swap local search ---
        # Try to improve by replacing one selected node with another to enable adding more
        selected_set = set(greedy_result)
        selected_list = list(greedy_result)
        improved = True
        max_iters = 500
        iter_count = 0

        while improved and not is_deadline() and iter_count < max_iters:
            improved = False
            iter_count += 1
            current_size = len(selected_set)

            # For each selected node, try removing it and see if we can add 2+ more
            for i in range(len(selected_list)):
                if is_deadline():
                    break
                remove_node = selected_list[i]
                # Tentatively remove
                selected_set.remove(remove_node)

                # Find all nodes that can now be added (not in selected, not conflicting with selected)
                addable = []
                for j in range(n):
                    if j in selected_set or j == remove_node:
                        continue
                    # Check if j conflicts with any currently selected
                    can_add = True
                    for s in selected_set:
                        if conflict[j, s]:
                            can_add = False
                            break
                    if can_add:
                        addable.append(j)

                # From addable, greedily add as many as possible
                added = []
                remaining_add = set(addable)
                while remaining_add:
                    # Pick one with minimum conflicts among remaining_add
                    best_add = None
                    best_conf = 999999
                    for a in remaining_add:
                        conf_count = sum(1 for b in added if conflict[a, b])
                        if conf_count < best_conf:
                            best_conf = conf_count
                            best_add = a
                        if conf_count > 0:
                            continue
                    if best_add is None or best_conf > 0:
                        # Can't add any more without conflicts
                        break
                    added.append(best_add)
                    remaining_add.discard(best_add)
                    # Remove neighbors of best_add from remaining_add
                    nbs = np.where(conflict[best_add])[0]
                    for nb in nbs:
                        if nb in remaining_add:
                            remaining_add.discard(nb)

                # If we can add more than 1 (net gain > 0), do the swap
                if len(added) > 1:
                    for a in added:
                        selected_set.add(a)
                    selected_list = list(selected_set)
                    improved = True
                    break  # Restart the improvement loop

                # Undo the tentative removal
                selected_set.add(remove_node)

        result = V[sorted(selected_set)]
        return result

    # ------------------------------------------------------------------
    # 4) Try each shell independently with the new solver
    # ------------------------------------------------------------------
    # Shell m=2: 220 vectors, max_ip_allowed = 1 (since dist^2 = 4-2*ip >= 2 => ip <= 1)
    # Actually for m=2: dist^2 = 2+2-2*ip = 4-2*ip >= 2 => ip <= 1
    V2 = gen_shell(2)
    try:
        res2 = solve_shell(V2, 1)
        update_best(res2)
    except Exception:
        pass

    # Shell m=3: 1320 vectors, max_ip_allowed = 0 (dist^2 = 6-2*ip >= 3 => ip <= 1.5, so ip <= 1)
    # Wait: for same-norm vectors with norm^2=m, dist^2 = 2m - 2*ip. Need dist^2 >= m => 2m-2*ip >= m => ip <= m/2.
    # For m=3: ip <= 1.5, so ip <= 1 (integer)
    V3 = gen_shell(3)
    try:
        res3 = solve_shell(V3, 1)
        update_best(res3)
    except Exception:
        pass

    # Shell m=4: 5302 vectors, max_ip_allowed = 2 (ip <= 4/2 = 2)
    # This is large; the Gram matrix is 5302x5302 = 28M entries. Might be slow.
    # Let's try but with a time check
    if not is_deadline():
        V4 = gen_shell(4)
        if V4.shape[0] <= 6000:
            try:
                res4 = solve_shell(V4, 2)
                update_best(res4)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 5) Mixed-shell approach: combine m=2 and m=3
    #    If we use only m=2 vectors, max norm is sqrt(2), need all pairs dist >= sqrt(2).
    #    If we add m=3 vectors, max norm is sqrt(3), need ALL pairs dist >= sqrt(3).
    #    For two m=2 vectors: dist^2 = 4-2*ip >= 3 => ip <= 0.5 => ip <= 0.
    #    So we can only use m=2 vectors with pairwise ip <= 0.
    #    This is a much stricter constraint. Let's try.
    # ------------------------------------------------------------------
    if not is_deadline():
        # Build combined pool of m=2 and m=3 vectors
        combined = np.vstack([V2, V3]) if V3.shape[0] > 0 else V2
        n_combined = combined.shape[0]
        
        # For mixed shells, the constraint is: dist(u,v) >= max(||u||, ||v||)
        # If both are m=2: need ip <= 0
        # If both are m=3: need ip <= 1
        # If one m=2, one m=3: dist^2 = 5 - 2*ip >= 3 => ip <= 1
        
        # This is a heterogeneous constraint. Let's build the conflict matrix directly.
        if n_combined <= 2000:
            G = combined.astype(np.int64) @ combined.astype(np.int64).T
            norms2 = np.sum(combined.astype(np.int64)**2, axis=1)
            
            # Conflict: dist^2 < max(norm_u^2, norm_v^2)
            # dist^2 = norm_u^2 + norm_v^2 - 2*ip
            # Need: norm_u^2 + norm_v^2 - 2*ip >= max(norm_u^2, norm_v^2)
            # i.e., min(norm_u^2, norm_v^2) >= 2*ip
            
            conflict = np.zeros((n_combined, n_combined), dtype=bool)
            for i in range(n_combined):
                for j in range(i+1, n_combined):
                    min_norm2 = min(norms2[i], norms2[j])
                    if 2 * G[i, j] > min_norm2:
                        conflict[i, j] = True
                        conflict[j, i] = True
            
            # Minimum-degree greedy
            selected = set()
            remaining = set(range(n_combined))
            
            while remaining and not is_deadline():
                # Find min degree in remaining
                min_deg = 999999
                pick = -1
                for i in remaining:
                    deg = sum(1 for j in remaining if j != i and conflict[i, j])
                    if deg < min_deg:
                        min_deg = deg
                        pick = i
                if pick == -1:
                    break
                selected.add(pick)
                remaining.discard(pick)
                # Remove neighbors
                nbs = np.where(conflict[pick])[0]
                for nb in nbs:
                    if nb in remaining:
                        remaining.discard(nb)
            
            if selected:
                result = combined[sorted(selected)]
                update_best(result)

    # ------------------------------------------------------------------
    # 6) Return best
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
