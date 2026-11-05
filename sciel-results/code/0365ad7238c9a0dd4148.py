import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_tsubs)]

    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = tsub_index[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)

    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) if subs else np.array([], dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) if blocks else np.array([], dtype=np.int32) for blocks in subset_to_blocks]

    def run_greedy_full(random_seed=None):
        """
        Greedy set cover using NumPy for fast gain computation.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(num_tsubs, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)

        selected_blocks = []

        while True:
            if check_time(10.0):
                break
            if not np.any(uncovered):
                break

            # Add small noise to break ties and add diversity
            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            if len(b_subs) == 0:
                continue

            newly_covered = b_subs[uncovered[b_subs]]
            if len(newly_covered) == 0:
                continue

            uncovered[newly_covered] = False
            # Vectorized gain update
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                if len(blocks_covering) > 0:
                    gains[blocks_covering] -= 1

        return selected_blocks

    def run_greedy_partial(uncovered_indices, random_seed=None):
        """
        Greedy to cover only the given uncovered t-subsets.
        """
        if len(uncovered_indices) == 0:
            return []

        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            blocks_covering = subset_block_arrays[s_idx]
            if len(blocks_covering) > 0:
                gains[blocks_covering] += 1

        selected_blocks = []
        current_uncovered = set(uncovered_indices)

        while current_uncovered:
            if check_time(5.0):
                break

            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            if len(b_subs) == 0:
                continue

            newly_covered = [s for s in b_subs if s in current_uncovered]
            if not newly_covered:
                continue

            for s in newly_covered:
                current_uncovered.discard(s)
                blocks_covering = subset_block_arrays[s]
                if len(blocks_covering) > 0:
                    gains[blocks_covering] -= 1

        return selected_blocks

    def compute_coverage_counts(blocks_indices):
        """Compute coverage count for each t-subset."""
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in blocks_indices:
            b_subs = block_subset_arrays[b_idx]
            if len(b_subs) > 0:
                cover_count[b_subs] += 1
        return cover_count

    def minimize_with_deletion_and_swaps(blocks_indices, time_limit, random_seed=None):
        """
        Local search with deletion and swap operations.
        Returns (blocks, cover_count).
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks, np.zeros(num_tsubs, dtype=np.int32)

        cover_count = compute_coverage_counts(current_blocks)
        current_blocks_set = set(current_blocks)

        improved = True
        local_start = time.time()

        while improved and (time.time() - local_start) < time_limit and not check_time(5.0):
            improved = False

            # Deletion pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)

            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - local_start) > time_limit or check_time(3.0):
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                if len(b_subs) == 0:
                    # Empty block, always redundant
                    current_blocks.pop(i)
                    current_blocks_set.discard(b_idx)
                    improved = True
                    break

                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart scan after removal

            # Swap pass
            if (time.time() - local_start) > time_limit or check_time(3.0):
                break

            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)

            swaps_done = 0
            max_swaps = 20

            for i in indices:
                if (time.time() - local_start) > time_limit or check_time(2.0):
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue

                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                if len(old_subs) == 0:
                    continue

                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in old_subs:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)

                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block or new_block in current_blocks_set:
                        continue

                    # Check if new block covers all unique_subs (none here, so always ok)
                    # Perform swap
                    for s_idx in old_subs:
                        cover_count[s_idx] -= 1
                    new_subs = block_subset_arrays[new_block]
                    if len(new_subs) > 0:
                        cover_count[new_subs] += 1
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = []
                    for s_idx in unique_subs:
                        blocks_covering = subset_block_arrays[s_idx]
                        if len(blocks_covering) == 0:
                            candidates = []
                            break
                        if not candidates:
                            candidates = list(blocks_covering)
                        else:
                            candidates = [b for b in candidates if b in set(blocks_covering)]
                        if not candidates:
                            break

                    if candidates:
                        # Filter out current blocks and old_block
                        candidates = [b for b in candidates if b not in current_blocks_set and b != old_block]
                        if candidates:
                            # Pick candidate that maximizes coverage of non-unique subsets
                            best_candidate = None
                            best_score = -1
                            for cand in candidates[:20]:  # Limit candidates
                                cand_subs = block_subset_arrays[cand]
                                if len(cand_subs) == 0:
                                    score = 0
                                else:
                                    score = np.sum(cover_count[cand_subs] > 1)
                                if score > best_score:
                                    best_score = score
                                    best_candidate = cand

                            if best_candidate is not None:
                                # Perform swap
                                for s_idx in old_subs:
                                    cover_count[s_idx] -= 1
                                new_subs = block_subset_arrays[best_candidate]
                                if len(new_subs) > 0:
                                    cover_count[new_subs] += 1
                                current_blocks_set.discard(old_block)
                                current_blocks_set.add(best_candidate)
                                current_blocks[i] = best_candidate
                                swaps_done += 1

        return current_blocks, cover_count

    def repair_coverage(blocks_indices, missing_indices, time_limit, random_seed=None):
        """
        Try to add blocks to cover missing t-subsets.
        Returns list of added blocks, or None if failed.
        """
        if not missing_indices:
            return []

        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        added_blocks = []
        current_missing = list(missing_indices)
        current_blocks_set = set(blocks_indices)
        local_start = time.time()

        while current_missing and (time.time() - local_start) < time_limit and not check_time(3.0):
            # Find best block to add
            best_block = -1
            best_gain = 0

            # Sample candidates
            sample_size = min(500, num_blocks)
            candidates = rng.sample(range(num_blocks), sample_size)

            for b_idx in candidates:
                if b_idx in current_blocks_set:
                    continue
                b_subs = block_subset_arrays[b_idx]
                if len(b_subs) == 0:
                    continue
                gain = sum(1 for s in b_subs if s in set(current_missing))
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1 or best_gain == 0:
                break

            added_blocks.append(best_block)
            current_blocks_set.add(best_block)

            b_subs = block_subset_arrays[best_block]
            current_missing = [s for s in current_missing if s not in set(b_subs)]

        if current_missing:
            return None  # Failed to cover all missing
        return added_blocks

    # Main Execution
    # 1. Greedy Initialization
    initial = run_greedy_full(random_seed=42)

    if not initial:
        return []

    # 2. Minimize with deletion and swaps
    best_blocks, best_cover_count = minimize_with_deletion_and_swaps(initial, time_limit=30.0, random_seed=42)
    best_len = len(best_blocks)

    # 3. Multiple repair-based random restarts
    for seed in range(10):
        if check_time(15.0):
            break

        # Start from current best
        current_blocks = list(best_blocks)
        current_cover_count = best_cover_count.copy()
        current_blocks_set = set(current_blocks)

        rng = random.Random(seed * 100 + 1)
        local_start = time.time()
        max_iters = 500

        for it in range(max_iters):
            if check_time(10.0) or (time.time() - local_start) > 20.0:
                break

            if len(current_blocks) < 2:
                break

            # Pick a random block to remove
            remove_idx = rng.randint(0, len(current_blocks) - 1)
            removed_block = current_blocks[remove_idx]
            removed_subs = block_subset_arrays[removed_block]

            # Tentative removal
            new_blocks = [x for x in current_blocks if x != removed_block]
            new_blocks_set = set(new_blocks)

            # Calculate missing
            missing = []
            for s_idx in removed_subs:
                if current_cover_count[s_idx] == 1:
                    missing.append(s_idx)

            if not missing:
                # Redundant block found
                for s_idx in removed_subs:
                    current_cover_count[s_idx] -= 1
                current_blocks = new_blocks
                current_blocks_set = new_blocks_set
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
                    best_cover_count = current_cover_count.copy()
                continue

            # Try to repair
            added = repair_coverage(new_blocks, missing, time_limit=5.0, random_seed=seed * 1000 + it)

            if added is not None:
                new_size = len(new_blocks) + len(added)
                if new_size < len(current_blocks):
                    # Accept
                    for b_idx in added:
                        b_subs = block_subset_arrays[b_idx]
                        if len(b_subs) > 0:
                            current_cover_count[b_subs] += 1
                    current_blocks = new_blocks + added
                    current_blocks_set = set(current_blocks)
                    if new_size < best_len:
                        best_len = new_size
                        best_blocks = list(current_blocks)
                        best_cover_count = current_cover_count.copy()
                elif new_size == len(current_blocks) and rng.random() < 0.05:
                    # Accept neutral move
                    for b_idx in added:
                        b_subs = block_subset_arrays[b_idx]
                        if len(b_subs) > 0:
                            current_cover_count[b_subs] += 1
                    current_blocks = new_blocks + added
                    current_blocks_set = set(current_blocks)
            # If not repaired, reject

        # Final deletion pass on current best
        if not check_time(5.0):
            final_blocks, final_cover_count = minimize_with_deletion_and_swaps(best_blocks, time_limit=10.0, random_seed=seed + 100)
            if len(final_blocks) < best_len:
                best_len = len(final_blocks)
                best_blocks = final_blocks
                best_cover_count = final_cover_count

    # 4. Final verification and cleanup
    if not check_time(3.0):
        final_blocks, final_cover_count = minimize_with_deletion_and_swaps(best_blocks, time_limit=5.0, random_seed=999)
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks

    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_blocks]
