import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 570  # Safety margin (600s limit, leave 30s buffer)

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)

    # Build conflict masks: conflict_mask[i] has bit j set iff i conflicts with j
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    # Precompute degree (number of conflicts) for each vertex
    degree = [bin(conflict_mask[i]).count('1') for i in range(m)]

    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def greedy_construct_weighted():
        """Weighted greedy: at each step pick the available candidate with lowest degree."""
        selected_mask = 0
        selected_list = []
        available = list(range(m))
        while available:
            best_idx = 0
            best_deg = degree[available[0]]
            for i in range(1, len(available)):
                if degree[available[i]] < best_deg:
                    best_idx = i
                    best_deg = degree[available[i]]
            candidate = available.pop(best_idx)
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
                cm = conflict_mask[candidate]
                available = [v for v in available if (cm & (1 << v)) == 0]
        return selected_list, selected_mask

    def greedy_refill(mask):
        """Fast greedy: add all non-conflicting vertices to mask. Returns (new_mask, new_count_added)."""
        # Precompute for each vertex v: conflict_mask[v] & mask == 0 means v can be added
        # We iterate and add greedily
        added = 0
        for v in range(m):
            if (mask & (1 << v)) == 0 and (conflict_mask[v] & mask) == 0:
                mask |= (1 << v)
                added += 1
        return mask, added

    def local_search(subset, mask, max_rounds=50):
        """
        Local search: try to increase size by removing elements and greedily adding back.
        """
        subset = list(subset)
        mask = mask
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            n = len(subset)

            # Try removing 1 element and greedily adding back
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                trial_mask = mask & ~(1 << u)
                new_mask, added = greedy_refill(trial_mask)
                new_size = n - 1 + added
                if new_size > n:
                    # Reconstruct subset from new_mask
                    subset = [v for v in range(m) if (new_mask & (1 << v))]
                    mask = new_mask
                    improved = True
                    break

            # Try removing 2 elements and greedily adding back
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5:
                        return subset, mask
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            return subset, mask
                        u, v = subset[i], subset[j]
                        trial_mask = mask & ~(1 << u) & ~(1 << v)
                        new_mask, added = greedy_refill(trial_mask)
                        new_size = n - 2 + added
                        if new_size > n:
                            subset = [v2 for v2 in range(m) if (new_mask & (1 << v2))]
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break

        return subset, mask

    best_subset = []
    best_size = 0
    best_mask = 0

    random.seed(42)

    # Phase 1: Weighted greedy (deterministic)
    subset, mask = greedy_construct_weighted()
    subset, mask = local_search(subset, mask)
    if len(subset) > best_size:
        best_size = len(subset)
        best_subset = subset
        best_mask = mask

    # Phase 2: Many random greedy trials with local search
    for trial in range(50000):
        if time.time() > deadline - 15:
            break

        # Vary the strategy
        if trial % 4 == 0:
            order = list(range(m))
            random.shuffle(order)
        elif trial % 4 == 1:
            # Degree-ordered with random tie-breaking
            order = sorted(range(m), key=lambda x: (degree[x], random.random()))
        elif trial % 4 == 2:
            # Inverse degree (prefer high-degree vertices early to block them)
            order = sorted(range(m), key=lambda x: (-degree[x], random.random()))
        else:
            # Weighted random: lower degree vertices more likely to appear early
            weights = [1.0 / (degree[x] + 1) for x in range(m)]
            total_w = sum(weights)
            order = []
            remaining = list(range(m))
            while remaining:
                r = random.random() * sum(weights[i] for i in remaining)
                cum = 0
                for idx, v in enumerate(remaining):
                    cum += weights[v]
                    if cum >= r:
                        order.append(v)
                        remaining.pop(idx)
                        break

        subset, mask = greedy_construct(order)
        subset, mask = local_search(subset, mask)

        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
            best_mask = mask

    # Phase 3: Final polish - more aggressive local search on best solution
    if time.time() < deadline - 20 and best_subset:
        # Run local search with more rounds
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            old_size = len(best_subset)
            best_subset, best_mask = local_search(best_subset, best_mask, max_rounds=200)
            if len(best_subset) == old_size:
                break

    # Final validation
    if not best_subset:
        subset, mask = greedy_construct(list(range(m)))
        best_subset = subset
        best_size = len(best_subset)

    # Verify no forbidden pairs
    best_set = set(best_subset)
    for a in best_subset:
        for b in best_subset:
            if a == b:
                continue
            diff = (a - b) % m
            if diff in forbidden:
                # Should not happen, but if it does, remove b
                best_subset.remove(b)
                best_set.discard(b)
                break

    return np.array(best_subset, dtype=np.int64)
