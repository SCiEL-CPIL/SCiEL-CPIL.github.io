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
            # Pick candidate with minimum degree among available
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
                # Remove all conflicts from available
                cm = conflict_mask[candidate]
                available = [v for v in available if (cm & (1 << v)) == 0]
        return selected_list, selected_mask

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
                trial_subset = subset[:i] + subset[i+1:]

                new_mask = trial_mask
                new_subset = list(trial_subset)
                for v in range(m):
                    if (new_mask & (1 << v)) == 0 and (conflict_mask[v] & new_mask) == 0:
                        new_mask |= (1 << v)
                        new_subset.append(v)

                if len(new_subset) > len(subset):
                    subset = new_subset
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
                        trial_subset = [x for x in subset if x != u and x != v]

                        new_mask = trial_mask
                        new_subset = list(trial_subset)
                        for w in range(m):
                            if (new_mask & (1 << w)) == 0 and (conflict_mask[w] & new_mask) == 0:
                                new_mask |= (1 << w)
                                new_subset.append(w)

                        if len(new_subset) > len(subset):
                            subset = new_subset
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
    for trial in range(20000):
        if time.time() > deadline - 15:
            break

        # Vary the strategy: sometimes shuffle, sometimes use degree-ordered with noise
        if trial % 3 == 0:
            order = list(range(m))
            random.shuffle(order)
        elif trial % 3 == 1:
            # Degree-ordered with random tie-breaking
            order = sorted(range(m), key=lambda x: (degree[x], random.random()))
        else:
            # Inverse degree (prefer high-degree vertices early to block them)
            order = sorted(range(m), key=lambda x: (-degree[x], random.random()))

        subset, mask = greedy_construct(order)
        subset, mask = local_search(subset, mask)

        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
            best_mask = mask

    # Phase 3: Final polish - more aggressive local search on best solution
    if time.time() < deadline - 20 and best_subset:
        # Run local search with more rounds
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            old_size = len(best_subset)
            best_subset, best_mask = local_search(best_subset, best_mask, max_rounds=100)
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
