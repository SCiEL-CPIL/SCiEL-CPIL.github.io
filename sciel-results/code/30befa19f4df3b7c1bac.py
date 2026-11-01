import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

    # Precompute all t-subsets and k-blocks
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build incidence: for each block, list of t-subset indices it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Precompute block index map for fast lookup
    block_idx_map = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    def randomized_greedy(seed, top_k=5, scarcity_weight=0.0):
        """Greedy with random tie-breaking and optional scarcity weighting."""
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        # Compute scarcity: number of blocks that cover each t-subset
        scarcity = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_idx in range(num_t_subsets):
            scarcity[ts_idx] = len(ts_to_blocks[ts_idx])

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            # Find all blocks with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            # Apply scarcity weighting if requested
            if scarcity_weight > 0 and len(candidates) > 1:
                # Score each candidate by gain + scarcity_weight * avg scarcity of uncovered ts it covers
                scores = np.zeros(len(candidates), dtype=np.float64)
                for ci, b_idx in enumerate(candidates):
                    avg_scarcity = 0.0
                    count = 0
                    for ts_idx in block_to_ts[b_idx]:
                        if is_uncovered[ts_idx]:
                            avg_scarcity += 1.0 / max(scarcity[ts_idx], 1)
                            count += 1
                    scores[ci] = gain[b_idx] + scarcity_weight * (avg_scarcity / max(count, 1))
                # Pick from top candidates
                sorted_idx = np.argsort(-scores)
                pick_from = min(top_k, len(sorted_idx))
                b_idx = candidates[sorted_idx[random.randint(0, pick_from - 1)]]
            else:
                # Pick randomly from top-k candidates
                pick_from = min(top_k, len(candidates))
                if pick_from < len(candidates):
                    chosen = random.sample(list(candidates), pick_from)
                    b_idx = chosen[random.randint(0, pick_from - 1)]
                else:
                    b_idx = candidates[random.randint(0, len(candidates) - 1)]

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    def iterative_removal(blocks):
        """Iteratively remove blocks that are fully redundant."""
        if not blocks:
            return blocks

        # Build cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        changed = True
        while changed:
            if time.time() > deadline - 3.0:
                break
            changed = False
            # Scan in reverse to avoid index issues
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 3.0:
                    break
                # Check if block i is redundant
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    ts_idx = ts_index_map[ts]
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blocks[i], t):
                        cover_count[ts_index_map[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break

        return blocks

    def local_search_2exchange(blocks, max_iters=200):
        """Try 1-exchange (swap) and 2-exchange (remove one, add another) operations."""
        if not blocks:
            return blocks, False

        # Build cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        sol_tuples = set(tuple(b) for b in blocks)

        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break

            i = random.randint(0, len(blocks) - 1)
            # Find unique t-subsets of block i
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                ts_idx = ts_index_map[ts]
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                # Block is fully redundant, just remove it
                for ts in itertools.combinations(blocks[i], t):
                    cover_count[ts_index_map[ts]] -= 1
                blocks.pop(i)
                improved = True
                break

            # Find a block that covers all unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break

            if not candidates:
                continue

            # Filter out current block and already-in-solution blocks
            current_idx = block_idx_map.get(tuple(blocks[i]), -1)
            valid = [c for c in candidates
                     if c != current_idx and tuple(all_k_blocks[c]) not in sol_tuples]
            if not valid:
                continue

            new_idx = random.choice(valid)
            new_block = list(all_k_blocks[new_idx])

            # Perform swap
            for ts in itertools.combinations(blocks[i], t):
                cover_count[ts_index_map[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1

            blocks[i] = new_block
            sol_tuples.discard(tuple(new_block))
            sol_tuples.add(tuple(new_block))
            improved = True
            break

        return blocks, improved

    def local_search_remove_add(blocks, max_iters=200):
        """Try to remove a block and add a different one that covers its unique subsets."""
        if not blocks:
            return blocks, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        sol_tuples = set(tuple(b) for b in blocks)
        improved = False

        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break

            # Try removing a random block and see if we can cover its unique subsets with one new block
            i = random.randint(0, len(blocks) - 1)
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                ts_idx = ts_index_map[ts]
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                # Fully redundant, remove it
                for ts in itertools.combinations(blocks[i], t):
                    cover_count[ts_index_map[ts]] -= 1
                blocks.pop(i)
                improved = True
                break

            # Find a block that covers all unique_ts and is not in solution
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break

            if not candidates:
                continue

            current_idx = block_idx_map.get(tuple(blocks[i]), -1)
            valid = [c for c in candidates
                     if c != current_idx and tuple(all_k_blocks[c]) not in sol_tuples]
            if not valid:
                continue

            new_idx = random.choice(valid)
            new_block = list(all_k_blocks[new_idx])

            # Remove old block
            for ts in itertools.combinations(blocks[i], t):
                cover_count[ts_index_map[ts]] -= 1
            blocks.pop(i)
            sol_tuples.discard(tuple(new_block))

            # Add new block
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1
            blocks.append(new_block)
            sol_tuples.add(tuple(new_block))

            improved = True
            break

        return blocks, improved

    # Main search
    best_solution = None
    best_len = float('inf')

    # Phase 1: Multiple randomized greedy constructions with varying parameters
    num_constructions = 40
    for i in range(num_constructions):
        if time.time() > deadline - 15.0:
            break
        top_k = random.choice([3, 5, 8, 10, 15])
        scarcity_weight = random.choice([0.0, 0.1, 0.5, 1.0])
        sol = randomized_greedy(i, top_k=top_k, scarcity_weight=scarcity_weight)
        if sol is not None:
            sol = iterative_removal(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    if best_solution is None:
        # Fallback: deterministic greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 10.0:
                break
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            b_idx = int(np.argmax(gain))
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        best_solution = sol
        best_len = len(sol)

    # Phase 2: Local search - alternate between 1-exchange and remove-add
    for iteration in range(100):
        if time.time() > deadline - 8.0:
            break

        # Try 1-exchange
        best_solution, improved = local_search_2exchange(best_solution, max_iters=150)
        if improved:
            best_solution = iterative_removal(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)

        if time.time() > deadline - 5.0:
            break

        # Try remove-add
        best_solution, improved2 = local_search_remove_add(best_solution, max_iters=150)
        if improved2:
            best_solution = iterative_removal(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)

        if not improved and not improved2:
            # No improvement in either method, try a few more iterations with different random seeds
            for retry in range(5):
                if time.time() > deadline - 3.0:
                    break
                random.seed(retry * 1000 + iteration)
                best_solution, imp = local_search_2exchange(best_solution, max_iters=100)
                if imp:
                    best_solution = iterative_removal(best_solution)
                    if len(best_solution) < best_len:
                        best_len = len(best_solution)
                    break

    # Phase 3: Final cleanup with aggressive iterative removal
    for _ in range(30):
        if time.time() > deadline - 2.0:
            break
        new_sol = iterative_removal(list(best_solution))
        if len(new_sol) < len(best_solution):
            best_solution = new_sol
            best_len = len(best_solution)
        else:
            break

    return best_solution
