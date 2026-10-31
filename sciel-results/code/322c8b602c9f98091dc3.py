import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0])

    def greedy_mis():
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int64)
        rng = np.random.default_rng(random.randint(0, 2**31 - 1))

        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected.append(chosen)

            neighbors = neighbor_lists[chosen]
            to_remove_set = {chosen}
            for n in neighbors:
                if alive[n]:
                    to_remove_set.add(n)

            to_remove = np.array(list(to_remove_set))
            alive[to_remove] = False

            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
                degrees[remaining] -= removed_neighbor_counts

        return selected

    # Run multiple greedy passes and keep the best
    best_selected = []
    num_greedy = 20
    for _ in range(num_greedy):
        if time.time() > deadline - 30:
            break
        sel = greedy_mis()
        if len(sel) > len(best_selected):
            best_selected = sel

    # Local search improvement: try to add vertices that don't conflict with selected
    selected_set = set(best_selected)
    conflicts_with_selected = np.zeros(N4, dtype=bool)
    for s in best_selected:
        conflicts_with_selected[neighbor_lists[s]] = True
    conflicts_with_selected[list(best_selected)] = True

    # Try adding unselected vertices that don't conflict with any selected
    unselected = np.where(~conflicts_with_selected)[0]
    for u in unselected:
        if time.time() > deadline - 20:
            break
        # Check if u conflicts with any already selected
        if not conflicts_with_selected[u]:
            best_selected.append(u)
            conflicts_with_selected[u] = True
            conflicts_with_selected[neighbor_lists[u]] = True

    # Swap-based local search: try removing one selected and adding up to 2 unselected
    for iteration in range(50):
        if time.time() > deadline - 15:
            break

        selected_set = set(best_selected)
        # Build conflict mask for current selection
        conflict_mask = np.zeros(N4, dtype=bool)
        for s in best_selected:
            conflict_mask[neighbor_lists[s]] = True
        conflict_mask[list(best_selected)] = True

        # Find candidates to add
        add_candidates = np.where(~conflict_mask)[0]
        if len(add_candidates) == 0:
            break

        # Try single swap: remove one, add one
        improved = False
        for s in best_selected:
            if time.time() > deadline - 15:
                break

            # Temporarily remove s
            new_conflict_mask = np.zeros(N4, dtype=bool)
            for t in best_selected:
                if t == s:
                    continue
                new_conflict_mask[neighbor_lists[t]] = True
            new_conflict_mask[list(best_selected)] = True
            new_conflict_mask[s] = True  # s is no longer selected

            # Find what can be added
            can_add = np.where(~new_conflict_mask)[0]
            can_add = can_add[can_add != s]

            if len(can_add) > 0:
                # Try adding one
                for a in can_add[:5]:  # Try a few
                    new_selected = [t for t in best_selected if t != s] + [a]
                    if len(new_selected) > len(best_selected):
                        best_selected = new_selected
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    result_v4 = vectors_v4[best_selected]

    # V1 vectors: check compatibility with selected V4
    # V1 has norm^2 = 4, V4 has norm^2 = 4
    # dot(V1, V4): V1 has one entry ±2, V4 has four entries ±1
    # If V1's nonzero position is among V4's nonzero positions, dot = ±2
    # Otherwise dot = 0
    # So dot ≤ 2 always holds. All V1 are compatible with all V4.
    # V1 vs V1: two V1 vectors either are same (dot=4>2, conflict) or different positions (dot=0) or opposite sign same position (dot=-4)
    # So we can take at most one V1 per position (either +2 or -2), but not both.
    # Actually, same position same sign: dot = 4 > 2, conflict
    # Same position opposite sign: dot = -4, OK
    # Different positions: dot = 0, OK
    # So we can take all 22 V1 vectors! Wait no: (2,0,...) and (2,0,...) are the same vector, we only have one of each.
    # We have 11 positions × 2 signs = 22 vectors. Two vectors conflict if they're at the same position with same sign.
    # But we only have one vector per (position, sign) pair, so no two V1 vectors are identical.
    # Two V1 vectors at same position: one is +2, other is -2, dot = -4 ≤ 2, OK.
    # Two V1 vectors at different positions: dot = 0 ≤ 2, OK.
    # So all 22 V1 vectors are mutually compatible!

    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
