import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}

    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, store which tsets it covers (as a list of indices)
    block_tsets = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tset_index[ts])
        block_tsets.append(covered)

    # Reverse index: for each tset, which blocks cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_idx in ts_list:
            tset_to_blocks[ts_idx].append(b_idx)

    full_mask = (1 << num_tsets) - 1

    # Block bitmasks for fast intersection
    block_masks = []
    for ts_list in block_tsets:
        mask = 0
        for idx in ts_list:
            mask |= (1 << idx)
        block_masks.append(mask)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        used = [False] * num_blocks

        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1

            # Randomize order to get different solutions
            order = list(range(num_blocks))
            rng.shuffle(order)

            for i in order:
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if count >= 100:  # Good enough, don't need perfect
                        break

            if best_idx == -1 or best_count == 0:
                break

            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True

        cover_counts = [0] * num_tsets

        def update_counts(block_idx, delta):
            for ts_idx in block_tsets[block_idx]:
                cover_counts[ts_idx] += delta

        for b in sol:
            update_counts(b, 1)

        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 30

        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False

            # Phase 1: Remove redundant blocks
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                redundant = True
                for ts_idx in block_tsets[b]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break

                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break

            if improved:
                no_improve = 0
                continue

            # Phase 2: 1-for-1 swaps
            if sol:
                found_swap = False
                for _ in range(15):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    # Find critical tsets (only covered by b_out)
                    critical = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical.append(ts_idx)

                    if not critical:
                        # Redundant, remove it
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    # Find a block that covers all critical tsets
                    # Start with the rarest critical tset
                    critical_sorted = sorted(critical, key=lambda x: len(tset_to_blocks[x]))
                    candidates = set(tset_to_blocks[critical_sorted[0]])
                    for ts_idx in critical_sorted[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts_idx])

                    valid_candidates = [c for c in candidates if not in_sol[c]]
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break

            no_improve += 1

        return best_sol

    def lagrangian_guided(seed):
        """
        Uses Lagrangian relaxation: iteratively update weights for uncovered tsets,
        then do a weighted greedy to select blocks.
        """
        rng = random.Random(seed)
        weights = [1.0] * num_tsets  # Penalty weights for each tset
        best_sol = None
        best_size = float('inf')

        num_iterations = 15
        for it in range(num_iterations):
            if elapsed() > deadline - 10:
                break

            # Weighted greedy: select blocks by max (sum of weights of newly covered tsets)
            uncovered = [True] * num_tsets
            selected = []
            used = [False] * num_blocks

            for _ in range(400):  # Limit selections
                if elapsed() > deadline - 10:
                    break

                best_idx = -1
                best_score = -1

                # Sample a subset of blocks for speed
                sample_size = min(num_blocks, 20000)
                sample = rng.sample(range(num_blocks), sample_size)

                for i in sample:
                    if used[i]:
                        continue
                    score = 0.0
                    for ts_idx in block_tsets[i]:
                        if uncovered[ts_idx]:
                            score += weights[ts_idx]
                    if score > best_score:
                        best_score = score
                        best_idx = i

                if best_idx == -1 or best_score <= 0:
                    break

                used[best_idx] = True
                selected.append(best_idx)
                for ts_idx in block_tsets[best_idx]:
                    uncovered[ts_idx] = False

            # Check if we have a valid cover
            cover_counts = [0] * num_tsets
            for b in selected:
                for ts_idx in block_tsets[b]:
                    cover_counts[ts_idx] += 1

            valid = all(c >= 1 for c in cover_counts)
            if valid and len(selected) < best_size:
                best_size = len(selected)
                best_sol = list(selected)

            # Update weights: increase for uncovered tsets
            for ts_idx in range(num_tsets):
                if cover_counts[ts_idx] == 0:
                    weights[ts_idx] *= 1.5
                else:
                    weights[ts_idx] *= 0.95  # Decay slightly

            # Normalize weights to prevent overflow
            max_w = max(weights) if weights else 1.0
            if max_w > 1e6:
                weights = [w / max_w for w in weights]

        if best_sol is None:
            # Fallback to plain greedy
            return greedy_cover(seed)
        return best_sol

    # Main loop: try multiple strategies
    best_solution = None
    best_size = float('inf')

    # First, try standard greedy + local search with a few seeds
    for seed in range(3):
        if elapsed() > deadline - 40:
            break
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    # Then, try Lagrangian-guided approach
    for seed in range(3, 8):
        if elapsed() > deadline - 20:
            break
        sol = lagrangian_guided(seed)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    # Use remaining time for more local search on the best solution
    if best_solution is not None and elapsed() < deadline - 5:
        # Try more local search iterations on the best solution
        for seed in range(10, 15):
            if elapsed() > deadline - 5:
                break
            ls_sol = local_search(best_solution, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    if best_solution is None:
        return []

    return [list(all_blocks[i]) for i in best_solution]
