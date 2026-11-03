"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Generate Weight 2, 3, and 4 vectors
    # Max norm squared is 4 (from Weight 4)
    # Condition: dist_sq[i,j] >= max(norm_sq[i], norm_sq[j])
    # For weight-4 vs weight-4: dist_sq >= 4  <=>  8 - 2*dot >= 4  <=>  dot <= 2
    # For weight-4 vs weight-3: dist_sq >= 4  <=>  7 - 2*dot >= 4  <=>  dot <= 1.5 -> dot <= 1
    # For weight-4 vs weight-2: dist_sq >= 4  <=>  6 - 2*dot >= 4  <=>  dot <= 1
    # For weight-3 vs weight-3: dist_sq >= 3  <=>  6 - 2*dot >= 3  <=>  dot <= 1.5 -> dot <= 1
    # For weight-3 vs weight-2: dist_sq >= 3  <=>  5 - 2*dot >= 3  <=>  dot <= 1
    # For weight-2 vs weight-2: dist_sq >= 2  <=>  4 - 2*dot >= 2  <=>  dot <= 1

    candidates = []
    norms_sq = []

    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            candidates.append(v)
            norms_sq.append(2)

    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)

    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)

    # Compute dot products
    dots = np.dot(candidates, candidates.T)

    # Determine conflicts: dist_sq[i,j] < max(norm_sq[i], norm_sq[j])
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dots[i,j]
    # Conflict if dist_sq[i,j] < max(norms_sq[i], norms_sq[j])
    max_norm = np.maximum(norms_sq[:, None], norms_sq[None, :])
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < max_norm
    np.fill_diagonal(conflicts, False)

    # Precompute neighbors as sorted arrays for fast membership and iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(n_candidates)]
    neighbor_sets = [set(n.tolist()) for n in neighbors]

    # Degree of each node
    degrees = np.array([len(n) for n in neighbors])

    def grasp_construct(rng, time_limit, alpha=0.5):
        """
        GRASP construction: build an independent set using restricted candidate list.
        Heuristic: prefer candidates with lower degree among feasible ones.
        """
        selected = set()
        in_selected = np.zeros(n_candidates, dtype=bool)

        while True:
            if time.time() > time_limit:
                break

            # Find all feasible candidates
            if selected:
                conflict_mask = np.zeros(n_candidates, dtype=bool)
                for s in selected:
                    conflict_mask[neighbors[s]] = True
                conflict_mask[list(selected)] = True
                feasible = np.where(~conflict_mask)[0]
            else:
                feasible = np.arange(n_candidates)

            if len(feasible) == 0:
                break

            # Score: lower degree is better
            scores = -degrees[feasible].astype(np.float64)
            scores += rng.random(len(feasible)) * 0.1

            max_score = scores.max()
            min_score = scores.min()

            if max_score - min_score < 1e-10:
                rcl = feasible
            else:
                threshold = max_score - alpha * (max_score - min_score)
                rcl = feasible[scores >= threshold]

            if len(rcl) == 0:
                rcl = feasible

            pick_idx = rng.randint(len(rcl))
            chosen = rcl[pick_idx]

            selected.add(chosen)
            in_selected[chosen] = True

        return list(selected)

    def swap_local_search(sel_list, rng, time_limit, max_iters=200):
        """
        1-for-k swap local search: remove one element, greedily add back
        multiple elements that become feasible.
        """
        selected = set(sel_list)
        in_selected = np.zeros(n_candidates, dtype=bool)
        for s in selected:
            in_selected[s] = True

        best_size = len(selected)
        improved = True
        iters = 0

        while improved and time.time() < time_limit and iters < max_iters:
            improved = False
            iters += 1

            # Try removing each element and adding back as many as possible
            for s in list(selected):
                if time.time() > time_limit:
                    break

                # Remove s
                selected.remove(s)
                in_selected[s] = False

                # Find candidates that conflict with s, are not in selected,
                # and don't conflict with any remaining selected
                candidates_to_try = []
                for c in neighbors[s]:
                    if in_selected[c]:
                        continue
                    # Check if c conflicts with any remaining selected
                    if not (neighbor_sets[c] & selected):
                        candidates_to_try.append(c)

                # Also try all non-selected, non-neighbor candidates that are now feasible
                # (they may have been blocked indirectly)
                for c in range(n_candidates):
                    if in_selected[c] or c == s:
                        continue
                    if not (neighbor_sets[c] & selected):
                        if c not in candidates_to_try:
                            candidates_to_try.append(c)

                # Sort by degree (prefer lower degree)
                candidates_to_try.sort(key=lambda x: degrees[x])

                # Greedily add as many as possible
                added = 0
                for c in candidates_to_try:
                    if time.time() > time_limit:
                        break
                    if not (neighbor_sets[c] & selected):
                        selected.add(c)
                        in_selected[c] = True
                        added += 1

                new_size = len(selected)

                if new_size > best_size:
                    best_size = new_size
                    improved = True
                else:
                    # Restore s
                    selected.add(s)
                    in_selected[s] = True

        return list(selected)

    def grasp_iterate(rng, time_limit, alpha=0.5):
        """One GRASP iteration: construct + local search."""
        sol = grasp_construct(rng, time_limit, alpha)
        sol = swap_local_search(sol, rng, time_limit)
        return sol

    # Main loop
    best_selected = []
    rng = np.random.RandomState(42)
    random.seed(42)

    # Phase 1: Greedy baseline with min-degree ordering
    order = list(range(n_candidates))
    order.sort(key=lambda i: degrees[i])
    baseline = []
    in_baseline = set()
    for idx in order:
        if not (neighbor_sets[idx] & in_baseline):
            baseline.append(idx)
            in_baseline.add(idx)
    best_selected = baseline
    best_size = len(best_selected)

    # Phase 2: GRASP iterations
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_iterations = 0

    while time.time() < deadline - 5.0:
        n_iterations += 1
        alpha = alphas[n_iterations % len(alphas)]

        iter_time = deadline - 5.0 - time.time()
        if iter_time < 1.0:
            break

        # Split time: 30% for construction, rest for local search
        construct_time = time.time() + min(3.0, iter_time * 0.3)
        search_time = deadline - 5.0

        sol = grasp_iterate(rng, search_time, alpha)

        if len(sol) > best_size:
            best_size = len(sol)
            best_selected = sol

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    return candidates[best_selected].astype(np.int64)
# EVOLVE-BLOCK-END
