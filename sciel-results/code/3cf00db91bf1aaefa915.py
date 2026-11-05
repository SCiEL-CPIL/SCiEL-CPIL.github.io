import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)

    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)

    # Convert to numpy arrays for faster operations
    block_subsets_np = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_blocks_np = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]

    # Precompute set version for fast membership
    block_subsets_set = [set(subs) for subs in block_to_subsets]
    subset_blocks_set = [set(blocks) for blocks in subset_to_blocks]

    def run_greedy_full(random_seed=None, initial_blocks=None):
        """Greedy set cover with optional initial blocks and randomized tie-breaking."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy()

        selected_blocks = []

        # Start with initial blocks if provided
        if initial_blocks:
            for b_idx in initial_blocks:
                selected_blocks.append(b_idx)
                b_subs = block_subsets_np[b_idx]
                newly_covered_mask = uncovered[b_subs]
                newly_covered = b_subs[newly_covered_mask]
                for s in newly_covered:
                    uncovered[s] = False
                    for b in subset_blocks_np[s]:
                        gains[b] -= 1

        while True:
            if not np.any(uncovered):
                break

            # Add noise to break ties and explore
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            if best_idx in selected_blocks:
                # Skip if already selected
                gains[best_idx] = -1  # Mark as unavailable
                continue

            selected_blocks.append(best_idx)

            b_subs = block_subsets_np[best_idx]
            newly_covered_mask = uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            for s in newly_covered:
                uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1

        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        """Greedy set cover on a specific uncovered set."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []

        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_blocks_np[s_idx]:
                gains[b_idx] += 1

        selected_blocks = []
        current_uncovered = uncovered_mask.copy()

        while True:
            if not np.any(current_uncovered):
                break

            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subsets_np[best_idx]
            newly_covered_mask = current_uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            for s in newly_covered:
                current_uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1

        return selected_blocks

    def algorithm_x_repair(uncovered_mask, random_seed=None, max_depth=6):
        """
        Algorithm X style repair for small uncovered sets.
        Uses backtracking with constraint propagation to find a minimal covering set.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        uncovered_indices = list(np.where(uncovered_mask)[0])
        if not uncovered_indices:
            return []

        if len(uncovered_indices) > 30:
            # Fall back to greedy for large uncovered sets
            return run_greedy_partial(uncovered_mask, random_seed)

        # For small sets, use a targeted search
        # Sort by number of covering blocks (most constrained first)
        uncovered_indices.sort(key=lambda s: len(subset_blocks_np[s]))

        def backtrack(remaining, depth, selected):
            if not remaining:
                return selected
            if depth >= max_depth:
                return None

            # Pick the most constrained subset
            s = remaining[0]
            rest = remaining[1:]

            # Try each block that covers s
            candidates = subset_blocks_np[s].copy()
            rng.shuffle(candidates)

            for b in candidates:
                b_subs_set = block_subsets_set[b]
                new_remaining = [x for x in rest if x not in b_subs_set]
                if len(new_remaining) >= len(rest):
                    continue  # Block doesn't help with rest
                result = backtrack(new_remaining, depth + 1, selected + [b])
                if result is not None:
                    return result
            return None

        result = backtrack(uncovered_indices, 0, [])
        if result is not None:
            return result
        return run_greedy_partial(uncovered_mask, random_seed)

    def minimize(blocks_indices, random_seed=None):
        """Remove redundant blocks iteratively."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        current_blocks = list(blocks_indices)

        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1

        # Repeatedly remove redundant blocks
        changed = True
        while changed:
            changed = False
            redundant_blocks = []

            # Random order for diversity
            order = list(range(len(current_blocks)))
            rng.shuffle(order)

            for i in order:
                b_idx = current_blocks[i]
                is_redundant = True
                for s_idx in block_subsets_np[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(i)

            if not redundant_blocks:
                break

            # Remove all redundant blocks at once
            for remove_pos in sorted(redundant_blocks, reverse=True):
                b_idx_to_remove = current_blocks[remove_pos]
                for s_idx in block_subsets_np[b_idx_to_remove]:
                    cover_count[s_idx] -= 1
                current_blocks.pop(remove_pos)
                changed = True

        return current_blocks

    def structured_swap_search(blocks_indices, random_seed=None, max_time=10.0):
        """
        Structured swap: for each block, find its uniquely-covered subsets.
        If non-empty, only swap with a block that covers all of them.
        This maintains validity and often enables deletions.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        start_time = time.time()
        current_blocks = list(blocks_indices)

        if len(current_blocks) < 2:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1

        best = list(current_blocks)
        best_len = len(current_blocks)

        current_set = set(current_blocks)

        while time.time() - start_time < max_time:
            # Deletion pass
            changed = True
            while changed:
                changed = False
                order = list(range(len(current_blocks)))
                rng.shuffle(order)
                for i in order:
                    if i >= len(current_blocks):
                        continue
                    b_idx = current_blocks[i]
                    is_redundant = True
                    for s_idx in block_subsets_np[b_idx]:
                        if cover_count[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s_idx in block_subsets_np[b_idx]:
                            cover_count[s_idx] -= 1
                        current_blocks.pop(i)
                        current_set.discard(b_idx)
                        changed = True
                        break

            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best = list(current_blocks)

            if not current_blocks:
                break

            # Swap pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            swaps_done = 0
            max_swaps = 20

            for i in indices:
                if time.time() - start_time > max_time:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue

                b_old = current_blocks[i]

                # Find uniquely covered subsets
                unique_subs = []
                for s_idx in block_subsets_np[b_old]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)

                if not unique_subs:
                    # Not critical: try a random swap to diversify
                    new_b = rng.randint(0, num_blocks - 1)
                    if new_b in current_set:
                        continue
                    # Perform swap
                    for s_idx in block_subsets_np[b_old]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_subsets_np[new_b]:
                        cover_count[s_idx] += 1
                    current_blocks[i] = new_b
                    current_set.discard(b_old)
                    current_set.add(new_b)
                    swaps_done += 1
                else:
                    # Find candidates that cover all unique_subs
                    candidates = None
                    for s in unique_subs:
                        if candidates is None:
                            candidates = set(subset_blocks_np[s])
                        else:
                            candidates.intersection_update(subset_blocks_np[s])
                            if not candidates:
                                break

                    if candidates:
                        candidates.discard(b_old)
                        if candidates:
                            new_b = rng.choice(list(candidates))
                            if new_b in current_set:
                                continue
                            # Perform swap
                            for s_idx in block_subsets_np[b_old]:
                                cover_count[s_idx] -= 1
                            for s_idx in block_subsets_np[new_b]:
                                cover_count[s_idx] += 1
                            current_blocks[i] = new_b
                            current_set.discard(b_old)
                            current_set.add(new_b)
                            swaps_done += 1

            # After swaps, do another deletion pass
            changed = True
            while changed:
                changed = False
                order = list(range(len(current_blocks)))
                rng.shuffle(order)
                for i in order:
                    if i >= len(current_blocks):
                        continue
                    b_idx = current_blocks[i]
                    is_redundant = True
                    for s_idx in block_subsets_np[b_idx]:
                        if cover_count[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s_idx in block_subsets_np[b_idx]:
                            cover_count[s_idx] -= 1
                        current_blocks.pop(i)
                        current_set.discard(b_idx)
                        changed = True
                        break

            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best = list(current_blocks)

        return best

    def simulated_annealing(blocks_indices, random_seed=None, max_time=15.0):
        """Simulated annealing with structured moves."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        start_time = time.time()
        current_blocks = list(blocks_indices)

        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1

        current_uncovered = np.sum(cover_count == 0)
        current_energy = current_uncovered * 1000 + len(current_blocks)

        best_blocks = list(current_blocks)
        best_energy = current_energy

        T = 5.0
        cooling_rate = 0.99
        min_T = 0.1

        step = 0
        while T > min_T:
            if time.time() - start_time > max_time:
                break

            move_type = rng.choice(['remove', 'add', 'swap'])
            new_blocks = list(current_blocks)
            new_cover_count = cover_count.copy()

            if move_type == 'remove' and len(current_blocks) > 1:
                idx = rng.randint(0, len(new_blocks) - 1)
                b_remove = new_blocks[idx]
                for s_idx in block_subsets_np[b_remove]:
                    new_cover_count[s_idx] -= 1
                new_blocks.pop(idx)

            elif move_type == 'add':
                new_b = rng.randint(0, num_blocks - 1)
                if new_b not in new_blocks:
                    for s_idx in block_subsets_np[new_b]:
                        new_cover_count[s_idx] += 1
                    new_blocks.append(new_b)
                else:
                    step += 1
                    T *= cooling_rate
                    continue

            elif move_type == 'swap' and len(current_blocks) > 0:
                idx = rng.randint(0, len(new_blocks) - 1)
                b_old = new_blocks[idx]
                new_b = rng.randint(0, num_blocks - 1)
                if new_b != b_old and new_b not in new_blocks:
                    for s_idx in block_subsets_np[b_old]:
                        new_cover_count[s_idx] -= 1
                    for s_idx in block_subsets_np[new_b]:
                        new_cover_count[s_idx] += 1
                    new_blocks[idx] = new_b
                else:
                    step += 1
                    T *= cooling_rate
                    continue

            new_uncovered = np.sum(new_cover_count == 0)
            new_energy = new_uncovered * 1000 + len(new_blocks)

            delta_e = new_energy - current_energy

            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_blocks = new_blocks
                cover_count = new_cover_count
                current_uncovered = new_uncovered
                current_energy = new_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = list(current_blocks)

            step += 1
            T *= cooling_rate

        return best_blocks

    # Main search
    best_blocks = None
    best_len = float('inf')

    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s

    trial = 0
    while time.time() < deadline:
        # Strategy 1: Filler pre-selection + Greedy + Minimize
        # Pick a few high-gain blocks as initial seed
        if trial < 5:
            # Deterministic filler: pick top blocks by size
            top_blocks = np.argsort(-block_sizes)[:3]
            initial = list(top_blocks)
        else:
            # Randomized filler
            rng_fill = np.random.default_rng(trial)
            top_idx = np.argsort(-block_sizes)[:10]
            initial = list(rng_fill.choice(top_idx, size=2, replace=False))

        greedy_res = run_greedy_full(trial, initial_blocks=initial)
        min_res = minimize(greedy_res, trial)

        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res

        # Strategy 2: Structured swap search on best
        if best_blocks is not None and len(best_blocks) > 5 and time.time() < deadline - 20:
            swap_res = structured_swap_search(best_blocks, trial + 10000, max_time=8.0)
            min_res_swap = minimize(swap_res, trial + 11000)

            if len(min_res_swap) < best_len:
                best_len = len(min_res_swap)
                best_blocks = min_res_swap

        # Strategy 3: Simulated annealing on best
        if best_blocks is not None and len(best_blocks) > 5 and time.time() < deadline - 25:
            sa_res = simulated_annealing(best_blocks, trial + 20000, max_time=12.0)
            min_res_sa = minimize(sa_res, trial + 21000)

            if len(min_res_sa) < best_len:
                best_len = len(min_res_sa)
                best_blocks = min_res_sa

        # Strategy 4: Perturb + Algorithm-X repair + Minimize
        if best_blocks is not None and len(best_blocks) > 5 and time.time() < deadline - 15:
            perturbed = list(best_blocks)
            num_remove = min(2, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.Random(trial + 30000).sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)

                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subsets_np[b_idx]:
                        cover_count_temp[s_idx] += 1

                uncovered_mask = (cover_count_temp == 0).astype(bool)

                # Use Algorithm-X repair if uncovered is small
                if np.sum(uncovered_mask) <= 30:
                    repair_res = algorithm_x_repair(uncovered_mask, trial + 31000)
                else:
                    repair_res = run_greedy_partial(uncovered_mask, trial + 31000)

                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)

                min_res2 = minimize(new_set, trial + 32000)

                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1

    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)

    # Final minimize pass
    best_blocks = minimize(best_blocks, 99999)

    return [list(all_blocks[i]) for i in best_blocks]
