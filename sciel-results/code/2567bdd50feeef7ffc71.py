import itertools
import time
import random
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # 3. Precompute coverage as sparse representation + numpy incidence matrix
    block_tsets = [[] for _ in range(num_k)]
    # Build incidence matrix: rows = blocks, cols = t-sets (0/1)
    # Use int8 to save memory; 75582 * 11628 * 1 byte ~ 878 MB - too much
    # Instead, store per-block tset lists and use numpy for batch operations
    for i, block in enumerate(k_blocks):
        ts_list = []
        for sub in itertools.combinations(block, t):
            idx = t_index_map[sub]
            ts_list.append(idx)
        block_tsets[i] = ts_list

    # Build a numpy array of tset indices per block (padded) for fast gain computation
    max_tsets_per_block = max(len(x) for x in block_tsets)
    # Actually for v=19,k=8,t=5: C(8,5)=56 t-sets per block
    # num_t = C(19,5) = 11628
    # We'll use a different approach: precompute for each block which t-sets it covers
    # and use numpy boolean arrays for the uncovered set

    # Represent uncovered as a numpy boolean array of size num_t
    # For gain computation: for each block, count how many of its t-sets are uncovered
    # Precompute block_tsets as numpy arrays
    block_tsets_np = np.array([block_tsets[i] for i in range(num_k)], dtype=np.int32)
    # Shape: (num_k, max_tsets_per_block) - but they're all same length (56)
    # Actually all blocks have exactly C(k,t) t-sets
    tsets_per_block = len(block_tsets[0])
    block_tsets_np = block_tsets_np.reshape(num_k, tsets_per_block)

    full_mask = (1 << num_t) - 1

    # 4. Build tset_to_blocks mapping for fast candidate filtering in local search
    tset_to_blocks = [[] for _ in range(num_t)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_idx in ts_list:
            tset_to_blocks[ts_idx].append(b_idx)

    # Precompute numpy arrays for vectorized gain computation
    # For each block, we need to count overlaps with uncovered t-sets
    # We'll use a numpy boolean array for uncovered and index into block_tsets_np

    def _greedy_pass(seed):
        """One greedy set cover pass with optional randomization and weighted scoring."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered_np = np.ones(num_t, dtype=np.int8)  # 1 if uncovered
        selected = []
        selected_set = set()

        while uncovered_np.any():
            if time.time() > deadline:
                return None

            # Vectorized gain computation
            # For each block, count how many of its t-sets are still uncovered
            # block_tsets_np has shape (num_k, tsets_per_block)
            # uncovered_np has shape (num_t,)
            # gains[b] = sum of uncovered_np[block_tsets_np[b, :]]
            gains = uncovered_np[block_tsets_np].sum(axis=1)  # shape (num_k,)

            # Zero out already-selected blocks
            if selected_set:
                gains[list(selected_set)] = 0

            # Weighted scoring: boost blocks that cover more high-priority (rare) t-sets
            # Use a simple heuristic: weight by 1/sqrt(coverage_count) for each t-set
            # But for speed, let's just use the basic gain first
            # For weighted version, we could precompute weights

            # Find best block
            best_idx = int(np.argmax(gains))
            best_count = int(gains[best_idx])

            if best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered_np[block_tsets_np[best_idx]] = 0

        return selected

    def _greedy_pass_weighted(seed):
        """Weighted greedy: t-sets with fewer covering blocks get higher priority."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered_np = np.ones(num_t, dtype=np.float32)  # weights for uncovered t-sets
        selected = []
        selected_set = set()

        # Initial weights: 1 / (number of blocks covering this t-set)
        # This makes rare t-sets more valuable
        weights = np.ones(num_t, dtype=np.float32)
        for ts_idx in range(num_t):
            n_covering = len(tset_to_blocks[ts_idx])
            if n_covering > 0:
                weights[ts_idx] = 1.0 / np.sqrt(n_covering)
            else:
                weights[ts_idx] = 1.0

        uncovered_weights = weights.copy()

        while uncovered_np.any():
            if time.time() > deadline:
                return None

            # Weighted gain: sum of weights of uncovered t-sets in each block
            gains = (uncovered_weights[block_tsets_np] * uncovered_np[block_tsets_np]).sum(axis=1)

            if selected_set:
                gains[list(selected_set)] = 0

            best_idx = int(np.argmax(gains))
            best_count = gains[best_idx]

            if best_count < 1e-10:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered_np[block_tsets_np[best_idx]] = 0

        return selected

    def _remove_redundant(selected_indices):
        """Remove redundant blocks from a valid covering using count-based check."""
        if not selected_indices:
            return selected_indices

        kept = list(selected_indices)
        changed = True

        while changed and time.time() < deadline:
            changed = False
            new_kept = []
            # Build coverage counts
            cover_counts = [0] * num_t
            for idx in kept:
                for ts_idx in block_tsets[idx]:
                    cover_counts[ts_idx] += 1

            for i in range(len(kept)):
                idx = kept[i]
                # Check if any tset in this block has count 1 (only covered by this block)
                redundant = True
                for ts_idx in block_tsets[idx]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_kept.append(idx)

            if changed:
                kept = new_kept

        return kept

    def _local_search(selected_indices, max_steps=2000):
        """Local search to minimize number of blocks using count-based coverage."""
        if time.time() > deadline:
            return selected_indices

        rng = random.Random()
        current = list(selected_indices)
        in_sol = [False] * num_k
        for b in current:
            in_sol[b] = True

        cover_counts = [0] * num_t

        def update_counts(block_idx, delta):
            for ts_idx in block_tsets[block_idx]:
                cover_counts[ts_idx] += delta

        for b in current:
            update_counts(b, 1)

        best_sol = list(current)
        best_len = len(current)
        no_improve = 0
        max_no_improve = 50

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break

            improved = False

            # Phase 1: Remove redundant blocks
            order = list(range(len(current)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(current):
                    continue
                b = current[pos]
                redundant = True
                for ts_idx in block_tsets[b]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    current.pop(pos)
                    improved = True
                    if len(current) < best_len:
                        best_len = len(current)
                        best_sol = list(current)
                    break

            if improved:
                no_improve = 0
                continue

            # Phase 2: 1-for-1 Swap
            if current:
                found_swap = False
                max_swap_attempts = 50
                if time.time() > deadline - 30:
                    max_swap_attempts = 20

                for _ in range(max_swap_attempts):
                    if not current:
                        break
                    pos = rng.randint(0, len(current) - 1)
                    b_out = current[pos]

                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            crit_ts_list.append(ts_idx)

                    if not crit_ts_list:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        current.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    # Find the critical tset with the fewest covering blocks
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len

                    # Check candidates from the shortest list
                    crit_set = set(crit_ts_list)
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        # Check if this block covers all critical t-sets
                        covers_all = all(ts in set(block_tsets[b_idx]) for ts in crit_ts_list)
                        if covers_all:
                            valid_candidates.append(b_idx)

                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        current[pos] = b_in
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    no_improve = 0
                    if len(current) < best_len:
                        best_len = len(current)
                        best_sol = list(current)
                    continue
            else:
                break

            no_improve += 1
            if no_improve > max_no_improve:
                break

        return best_sol

    def _polish(selected_indices):
        """Final polishing: repeated redundancy removal and local swaps."""
        if time.time() > deadline:
            return selected_indices

        result = _remove_redundant(selected_indices)

        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            result = _local_search(result, max_steps=1000)
            result = _remove_redundant(result)

        return result

    best_blocks = None
    best_len = float('inf')

    # Phase 1: Run multiple greedy passes (both standard and weighted)
    attempt = 0
    while time.time() < deadline - 25.0:
        if attempt % 2 == 0:
            selected = _greedy_pass(attempt)
        else:
            selected = _greedy_pass_weighted(attempt)
        if selected is None:
            break

        selected = _remove_redundant(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]

        if best_len <= 330:
            break

        attempt += 1

    # Phase 2: Polish the best solution found
    if best_blocks is not None and time.time() < deadline - 1.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]

        polished = _polish(best_indices)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = [list(k_blocks[i]) for i in polished]

    # Fallback if no solution found
    if best_blocks is None:
        uncovered_np = np.ones(num_t, dtype=np.int8)
        selected = []
        selected_set = set()
        while uncovered_np.any() and time.time() < deadline:
            gains = uncovered_np[block_tsets_np].sum(axis=1)
            if selected_set:
                gains[list(selected_set)] = 0
            best_idx = int(np.argmax(gains))
            best_count = int(gains[best_idx])
            if best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered_np[block_tsets_np[best_idx]] = 0

        if selected:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
