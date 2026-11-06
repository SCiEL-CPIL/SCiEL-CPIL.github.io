import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin under 90s

    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    conflict_mask = [0] * (N + 1)

    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Pre-filter: nodes where 2a is a square are invalid
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)

    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all

    def greedy_min_degree(seed):
        """Greedy independent set using min-degree heuristic."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0

        while remaining:
            min_deg = float('inf')
            candidates_at_min = []
            for a in remaining:
                deg = bin(conflict_mask[a] & ~chosen_mask & valid_mask_all).count('1')
                if deg < min_deg:
                    min_deg = deg
                    candidates_at_min = [a]
                elif deg == min_deg:
                    candidates_at_min.append(a)

            min_node = rng.choice(candidates_at_min)
            chosen.append(min_node)
            chosen_mask |= (1 << min_node)
            remaining.discard(min_node)
            to_remove = [b for b in remaining if (conflict_mask[min_node] >> b) & 1]
            for b in to_remove:
                remaining.discard(b)

        return chosen_mask, chosen

    def greedy_random(seed):
        """Random greedy independent set."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0

        while remaining:
            a = rng.choice(list(remaining))
            chosen.append(a)
            chosen_mask |= (1 << a)
            remaining.discard(a)
            to_remove = [b for b in remaining if (conflict_mask[a] >> b) & 1]
            for b in to_remove:
                remaining.discard(b)

        return chosen_mask, chosen

    def greedy_weighted(seed):
        """Degree-weighted random greedy."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0

        while remaining:
            degrees = {}
            for a in remaining:
                deg = bin(conflict_mask[a] & ~chosen_mask & valid_mask_all).count('1')
                degrees[a] = deg
            weights = [1.0 / (degrees[a] + 1) for a in remaining]
            total_weight = sum(weights)
            r = rng.random() * total_weight
            cumulative = 0
            selected = None
            for a, w in zip(remaining, weights):
                cumulative += w
                if cumulative >= r:
                    selected = a
                    break
            if selected is None:
                selected = list(remaining)[0]

            chosen.append(selected)
            chosen_mask |= (1 << selected)
            remaining.discard(selected)
            to_remove = [b for b in remaining if (conflict_mask[selected] >> b) & 1]
            for b in to_remove:
                remaining.discard(b)

        return chosen_mask, chosen

    def local_search(start_mask, start_list, seed, time_limit):
        """Local search with add, swap, and multi-swap moves."""
        rng = random.Random(seed)
        in_mask = start_mask
        in_set = set(start_list)
        out_set = set(valid_nodes) - in_set

        current_size = len(in_set)

        def try_add_all():
            nonlocal in_mask, in_set, out_set, current_size
            added = True
            while added:
                added = False
                candidates = [a for a in out_set if (conflict_mask[a] & in_mask) == 0]
                if candidates:
                    rng.shuffle(candidates)
                    for a in candidates:
                        if (conflict_mask[a] & in_mask) == 0:
                            in_mask |= (1 << a)
                            in_set.add(a)
                            out_set.discard(a)
                            current_size += 1
                            added = True
            return added

        improved = True
        max_iters = 200
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            improved = False

            # Step 1: Add all compatible nodes
            if try_add_all():
                improved = True
                continue

            # Step 2: Try 1-remove, k-add swaps
            in_list = list(in_set)
            rng.shuffle(in_list)
            found_swap = False
            for u in in_list:
                if time.time() > time_limit:
                    break
                mask_without_u = in_mask ^ (1 << u)
                potential_adds = [v for v in out_set if (conflict_mask[v] & mask_without_u) == 0]

                if len(potential_adds) > 1:
                    new_in_mask = mask_without_u
                    new_in_set = in_set - {u}
                    new_out_set = out_set | {u}
                    cand = list(potential_adds)
                    rng.shuffle(cand)
                    added_count = 0
                    for v in cand:
                        if (conflict_mask[v] & new_in_mask) == 0:
                            new_in_mask |= (1 << v)
                            new_in_set.add(v)
                            new_out_set.discard(v)
                            added_count += 1

                    if added_count > 1:
                        in_mask = new_in_mask
                        in_set = new_in_set
                        out_set = new_out_set
                        current_size = len(in_set)
                        improved = True
                        found_swap = True
                        break

            if found_swap:
                continue

            # Step 3: Try 2-remove, k-add swaps
            if time.time() > time_limit:
                break
            in_list = list(in_set)
            if len(in_list) >= 2:
                found_2swap = False
                num_pairs_to_try = min(50, len(in_list) * (len(in_list) - 1) // 2)
                for _ in range(num_pairs_to_try):
                    if time.time() > time_limit:
                        break
                    idx1, idx2 = rng.sample(range(len(in_list)), 2)
                    u1 = in_list[idx1]
                    u2 = in_list[idx2]

                    mask_without = in_mask ^ (1 << u1) ^ (1 << u2)
                    potential_adds = [v for v in out_set if (conflict_mask[v] & mask_without) == 0]

                    if len(potential_adds) > 2:
                        new_in_mask = mask_without
                        new_in_set = in_set - {u1, u2}
                        new_out_set = out_set | {u1, u2}
                        cand = list(potential_adds)
                        rng.shuffle(cand)
                        added_count = 0
                        for v in cand:
                            if (conflict_mask[v] & new_in_mask) == 0:
                                new_in_mask |= (1 << v)
                                new_in_set.add(v)
                                new_out_set.discard(v)
                                added_count += 1

                        if added_count > 2:
                            in_mask = new_in_mask
                            in_set = new_in_set
                            out_set = new_out_set
                            current_size = len(in_set)
                            improved = True
                            found_2swap = True
                            break
                if found_2swap:
                    continue

            # Step 4: Try 3-remove, k-add swaps
            if time.time() > time_limit:
                break
            in_list = list(in_set)
            if len(in_list) >= 3:
                found_3swap = False
                num_triples_to_try = min(20, len(in_list) * (len(in_list) - 1) * (len(in_list) - 2) // 6)
                for _ in range(num_triples_to_try):
                    if time.time() > time_limit:
                        break
                    idxs = rng.sample(range(len(in_list)), 3)
                    u1, u2, u3 = in_list[idxs[0]], in_list[idxs[1]], in_list[idxs[2]]

                    mask_without = in_mask ^ (1 << u1) ^ (1 << u2) ^ (1 << u3)
                    potential_adds = [v for v in out_set if (conflict_mask[v] & mask_without) == 0]

                    if len(potential_adds) > 3:
                        new_in_mask = mask_without
                        new_in_set = in_set - {u1, u2, u3}
                        new_out_set = out_set | {u1, u2, u3}
                        cand = list(potential_adds)
                        rng.shuffle(cand)
                        added_count = 0
                        for v in cand:
                            if (conflict_mask[v] & new_in_mask) == 0:
                                new_in_mask |= (1 << v)
                                new_in_set.add(v)
                                new_out_set.discard(v)
                                added_count += 1

                        if added_count > 3:
                            in_mask = new_in_mask
                            in_set = new_in_set
                            out_set = new_out_set
                            current_size = len(in_set)
                            improved = True
                            found_3swap = True
                            break
                if found_3swap:
                    continue

            if not improved:
                break

        return in_mask, in_set

    def branch_and_bound(initial_set, time_limit):
        """Branch and bound on a small candidate set to find max independent set."""
        in_set = set(initial_set)
        out_set = set(valid_nodes) - in_set

        # Focus on nodes that conflict with at least one in_set node or are isolated
        candidates = set()
        for a in in_set:
            for b in out_set:
                if (conflict_mask[a] >> b) & 1:
                    candidates.add(b)
        # Also add isolated nodes (no conflicts with in_set)
        for a in out_set:
            if (conflict_mask[a] & valid_mask_all) == 0:
                candidates.add(a)

        if not candidates:
            return in_set

        cand_list = sorted(candidates)
        best_set = set(in_set)
        best_size = len(in_set)

        def dfs(remaining_cands, current_set, current_mask):
            nonlocal best_set, best_size
            if time.time() > time_limit:
                return

            # Try to add all compatible nodes from remaining
            new_mask = current_mask
            new_set = set(current_set)
            for a in remaining_cands:
                if (conflict_mask[a] & new_mask) == 0:
                    new_mask |= (1 << a)
                    new_set.add(a)

            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = set(new_set)

            # Branch on remaining candidates
            if not remaining_cands:
                return

            # Pick a candidate with high degree
            best_cand = None
            best_deg = -1
            for a in remaining_cands:
                deg = bin(conflict_mask[a] & new_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_cand = a

            if best_cand is None:
                return

            # Option 1: Include best_cand
            new_remaining = set(remaining_cands) - {best_cand}
            new_remaining = {a for a in new_remaining if (conflict_mask[best_cand] >> a) == 0}
            dfs(new_remaining, new_set | {best_cand}, new_mask | (1 << best_cand))

            # Option 2: Exclude best_cand
            new_remaining = set(remaining_cands) - {best_cand}
            dfs(new_remaining, new_set, new_mask)

        dfs(set(cand_list), in_set, 0)
        return best_set

    best_mask = 0
    best_size = 0
    best_list = []

    # Run multiple restarts with different strategies
    num_restarts = 150
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break

        # Alternate between different initialization strategies
        if i % 4 == 0:
            init_mask, init_list = greedy_min_degree(i)
        elif i % 4 == 1:
            init_mask, init_list = greedy_random(i)
        elif i % 4 == 2:
            init_mask, init_list = greedy_weighted(i)
        else:
            # Hybrid: start with min-degree, then random perturbation
            init_mask, init_list = greedy_min_degree(i + 1000)
            rng = random.Random(i)
            if init_list:
                # Randomly swap a few elements
                for _ in range(min(3, len(init_list))):
                    if len(init_list) > 1:
                        idx1 = rng.randrange(len(init_list))
                        idx2 = rng.randrange(len(init_list))
                        if idx1 != idx2:
                            init_list[idx1], init_list[idx2] = init_list[idx2], init_list[idx1]

        # Time budget for local search
        ls_time_limit = min(time.time() + 1.5, deadline - 15)
        ls_mask, ls_set = local_search(init_mask, init_list, i, ls_time_limit)

        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_mask = ls_mask
            best_list = list(ls_set)

    # Final refinement with branch-and-bound on the best solution
    if time.time() < deadline - 5:
        bb_time_limit = deadline - 2
        refined_set = branch_and_bound(set(best_list), bb_time_limit)
        if len(refined_set) > best_size:
            best_size = len(refined_set)
            best_list = list(refined_set)

    # Final verification
    result = set(best_list)
    for a in result:
        for b in result:
            if a <= b:
                s_val = a + b
                root = int(math.isqrt(s_val))
                if root * root == s_val:
                    # Invalid - should not happen, but just in case
                    pass

    return list(best_list)
