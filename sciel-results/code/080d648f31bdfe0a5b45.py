import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate all 3-nonzero vectors with entries in {-1, 0, 1}
    # Norm squared = 3 for all
    candidates = []
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)  # C(11,3) * 8 = 165 * 8 = 1320

    # Compute dot product matrix
    M = candidates @ candidates.T

    # All have norm squared = 3
    # Distance squared = 3 + 3 - 2*dot = 6 - 2*dot
    # Need distance squared >= 3 (max norm squared)
    # 6 - 2*dot >= 3 => dot <= 1.5 => dot <= 1 (integer)
    # Conflict if dot > 1
    Conflict = (M > 1).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)

    # Also add 2-nonzero vectors (norm squared = 2)
    # These have smaller norm, so the constraint is distance >= max_norm
    # If we include them, max_norm might be 3 (from 3-nonzero) or 2 (if only 2-nonzero)
    # Let's focus on 3-nonzero only for simplicity, but also try 4-nonzero

    # Generate 4-nonzero vectors (norm squared = 4)
    candidates_4 = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates_4.append(v)

    candidates_4 = np.array(candidates_4, dtype=np.int32)
    N4 = len(candidates_4)  # C(11,4) * 16 = 330 * 16 = 5280

    # For 4-nonzero: norm squared = 4, need distance squared >= 4
    # 4 + 4 - 2*dot >= 4 => dot <= 2
    # Conflict if dot > 2

    # Strategy: Use support-based approach for 3-nonzero vectors
    # Group by support
    supports = {}
    for i, v in enumerate(candidates):
        support = tuple(j for j in range(D) if v[j] != 0)
        if support not in supports:
            supports[support] = []
        supports[support].append(i)

    n_supports = len(supports)  # 165
    support_list = list(supports.keys())
    support_to_idx = {s: i for i, s in enumerate(support_list)}

    # For each support, find max independent set of sign vectors
    # Within a support, conflict if dot > 1, i.e., dot in {2, 3}
    # dot = 2t - 3 where t = number of agreeing signs
    # dot > 1 means 2t - 3 > 1 => t > 2 => t = 3 (same signs)
    # Wait: dot = sum of products. If agree on t positions, dot = t - (3-t) = 2t-3
    # dot > 1: 2t-3 > 1 => t > 2 => t = 3
    # So within same support, conflict only if all 3 signs are the same!
    # That means we can pick all 7 vectors except one? No...
    # If two vectors have all 3 signs same, they're identical. So within same support,
    # all 8 vectors are distinct, and any two have dot in {-3, -1, 1, 3}
    # dot > 1 only when dot = 3, which means identical vectors.
    # So within same support, NO conflicts! We can pick all 8!

    # Wait, that can't be right. Let me recheck.
    # Vector a = (1,1,1), vector b = (1,1,-1). dot = 1+1-1 = 1. Compatible.
    # Vector a = (1,1,1), vector b = (1,-1,-1). dot = 1-1-1 = -1. Compatible.
    # Vector a = (1,1,1), vector b = (-1,-1,-1). dot = -3. Compatible (|dot|=3, but dot=-3 <= 1)
    # So indeed, within same support, max dot = 1 for distinct vectors. All 8 are compatible!

    # Now between different supports:
    # If supports intersect in 0 or 1 position: max |dot| <= 1, always compatible
    # If supports intersect in 2 positions: dot can be -2, 0, 2. Conflict if dot > 1, i.e., dot = 2
    # So conflict when the two vectors agree on both shared positions

    # Build conflict between supports
    # For supports S1, S2 with |S1 ∩ S2| = 2, conflict exists between some sign pairs
    # We need to find max set of (support, sign) pairs with no conflicts

    # This is a large problem. Let's use a different approach:
    # Since within each support all 8 are compatible, the problem reduces to:
    # For each pair of supports with intersection 2, we cannot have sign pairs that agree on both shared coords.

    # Let's build a bipartite-like structure and use greedy + local search

    # Actually, let's just use the full 1320-vector conflict graph and apply spectral + local search

    # Spectral approach: compute Fiedler vector to find dense clusters
    # Then use local search to refine

    def greedy_max_independent_set(conflict, n, rng):
        """Greedy algorithm for maximum independent set."""
        active = np.ones(n, dtype=bool)
        deg = conflict.sum(axis=1).astype(np.int64)
        selected = []

        while active.any():
            if time.time() > deadline - 10:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break

            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)

            selected.append(idx)

            neighbors_mask = conflict[idx].astype(bool)
            removed_mask = np.zeros(n, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True

            removed_indices = np.where(removed_mask)[0]
            if len(removed_indices) > 0:
                removal_counts = conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]

            active[removed_mask] = False

        return selected

    def local_search_2swap(selected_set, conflict, n, rng, max_iters=100):
        """Local search: try to replace one vertex with two non-conflicting vertices."""
        selected_list = list(selected_set)
        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            rng.shuffle(selected_list)

            for i in selected_list:
                if time.time() > deadline - 15:
                    break

                # Find vertices not in selected_set that conflict only with i
                neighbors_i = np.where(conflict[i])[0]
                valid_js = [j for j in neighbors_i if j not in selected_set]

                if len(valid_js) < 2:
                    continue

                # Check compatibility with rest of set
                compatible = []
                for j in valid_js:
                    ok = True
                    for k in selected_set:
                        if k == i:
                            continue
                        if conflict[j, k]:
                            ok = False
                            break
                    if ok:
                        compatible.append(j)

                if len(compatible) < 2:
                    continue

                # Find pair (j1, j2) in compatible with no conflict between them
                found = False
                for a in range(len(compatible)):
                    if found:
                        break
                    ja = compatible[a]
                    for b in range(a + 1, len(compatible)):
                        jb = compatible[b]
                        if conflict[ja, jb] == 0:
                            selected_set.remove(i)
                            selected_set.add(ja)
                            selected_set.add(jb)
                            improved = True
                            found = True
                            break

                if improved:
                    break

        return selected_set

    def simulated_annealing(initial_set, conflict, n, max_steps=20000):
        """Simulated annealing on independent set."""
        if time.time() > deadline - 20:
            return set(initial_set)

        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size

        T = 5.0
        cooling_rate = 0.99995
        rng = np.random.default_rng(42)
        all_vertices = np.arange(n)

        for step in range(max_steps):
            if time.time() > deadline - 15:
                break

            move_type = rng.integers(0, 3)
            new_set = set(current_set)

            if move_type == 0:  # Remove
                if not current_set:
                    continue
                v_to_remove = rng.choice(list(current_set))
                new_set.remove(v_to_remove)
                delta_size = -1

            elif move_type == 1:  # Add
                v_to_add = rng.choice(all_vertices)
                if v_to_add in current_set:
                    continue
                is_compatible = True
                for v in current_set:
                    if conflict[v_to_add, v]:
                        is_compatible = False
                        break
                if not is_compatible:
                    continue
                new_set.add(v_to_add)
                delta_size = 1

            else:  # Swap
                if not current_set:
                    continue
                v_in = rng.choice(list(current_set))
                v_out = rng.choice(all_vertices)
                if v_out in current_set:
                    continue
                is_compatible = True
                for v in current_set:
                    if v == v_in:
                        continue
                    if conflict[v_out, v]:
                        is_compatible = False
                        break
                if not is_compatible:
                    continue
                new_set.remove(v_in)
                new_set.add(v_out)
                delta_size = 0

            new_size = len(new_set)

            if new_size >= best_size:
                current_set = new_set
                current_size = new_size
                if new_size > best_size:
                    best_set = set(new_set)
                    best_size = new_size
            else:
                delta = -delta_size
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_set = new_set
                        current_size = new_size

            T *= cooling_rate
            if T < 1e-4:
                break

        return best_set

    # Phase 1: Multiple greedy restarts
    best_set = set()
    best_size = 0

    n_restarts = 30
    for seed in range(n_restarts):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(seed)
        result = greedy_max_independent_set(Conflict, N, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)

    # Phase 2: Local search
    if time.time() < deadline - 30 and best_set:
        rng_ls = np.random.default_rng(123)
        improved_set = local_search_2swap(best_set, Conflict, N, rng_ls, max_iters=200)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set

    # Phase 3: Simulated annealing
    if time.time() < deadline - 20 and best_set:
        sa_result = simulated_annealing(best_set, Conflict, N, max_steps=30000)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result

        # Another local search pass
        if time.time() < deadline - 10:
            rng_ls2 = np.random.default_rng(456)
            improved_set = local_search_2swap(best_set, Conflict, N, rng_ls2, max_iters=100)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set

    # Phase 4: More restarts if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        for seed in range(n_restarts, n_restarts + 40):
            if time.time() > deadline - 15:
                break
            rng = np.random.default_rng(seed)
            result = greedy_max_independent_set(Conflict, N, rng)
            if len(result) > best_size:
                best_size = len(result)
                best_set = set(result)
                if time.time() < deadline - 15:
                    rng_ls3 = np.random.default_rng(seed * 7 + 1)
                    improved_set = local_search_2swap(best_set, Conflict, N, rng_ls3, max_iters=50)
                    if len(improved_set) > best_size:
                        best_size = len(improved_set)
                        best_set = improved_set

    # Construct final vectors
    final_vectors = candidates[list(best_set)]

    if len(final_vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return final_vectors.astype(np.int64)

if __name__ == "__main__":
    v = construct()
    print(f"Found {len(v)} vectors")
