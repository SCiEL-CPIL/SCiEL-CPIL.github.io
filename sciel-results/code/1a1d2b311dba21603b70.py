import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each point, precompute which t-subsets it belongs to
    point_to_ts = []
    for p in range(v):
        ts_list = []
        for ts in all_t_subsets:
            if p in ts:
                ts_list.append(ts_index[ts])
        point_to_ts.append(ts_list)

    # For each pair of points, precompute which t-subsets contain both
    # Actually, let's use a different approach for density-driven generation

    # Track uncovered t-subsets
    uncovered = np.ones(num_t_subsets, dtype=bool)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)

    selected_blocks = []
    selected_set = set()

    def time_ok():
        return time.time() < deadline - 3.0

    def block_covers_uncovered(block):
        """Count how many uncovered t-subsets this block covers"""
        count = 0
        for ts in itertools.combinations(block, t):
            idx = ts_index[ts]
            if uncovered[idx]:
                count += 1
        return count

    def density_generate_block(uncovered_ts_list, rng):
        """
        Generate a block that maximizes coverage of uncovered t-subsets.
        Uses a greedy point-selection heuristic:
        1. Start with empty block
        2. At each step, add the point that maximizes new uncovered t-subsets covered
        """
        if not uncovered_ts_list:
            return None

        # Build a frequency map: for each point, how many uncovered t-subsets contain it
        point_freq = np.zeros(v, dtype=np.int32)
        for ts_idx in uncovered_ts_list:
            ts = all_t_subsets[ts_idx]
            for p in ts:
                point_freq[p] += 1

        # Greedy point selection
        block = []
        remaining_uncovered = set(uncovered_ts_list)
        points_used = set()

        for _ in range(k):
            best_point = -1
            best_gain = -1
            for p in range(v):
                if p in points_used:
                    continue
                # Count how many remaining uncovered t-subsets contain p
                gain = 0
                for ts_idx in remaining_uncovered:
                    if p in all_t_subsets[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p

            if best_point < 0 or best_gain <= 0:
                break

            block.append(best_point)
            points_used.add(best_point)

            # Update remaining uncovered: remove t-subsets that are now fully covered
            # A t-subset is covered if all its points are in block
            new_remaining = set()
            for ts_idx in remaining_uncovered:
                ts = all_t_subsets[ts_idx]
                if all(p in points_used for p in ts):
                    pass  # Now covered
                else:
                    new_remaining.add(ts_idx)
            remaining_uncovered = new_remaining

            if not remaining_uncovered:
                break

        if len(block) < k:
            # Fill with random points
            remaining_points = [p for p in range(v) if p not in points_used]
            rng.shuffle(remaining_points)
            for p in remaining_points:
                if len(block) >= k:
                    break
                block.append(p)

        block.sort()
        return tuple(block)

    def fast_density_block(uncovered_mask, rng):
        """
        Faster density-driven block generation using vectorized operations.
        uncovered_mask: boolean array of length num_t_subsets
        """
        # Count uncovered t-subsets per point
        # Each t-subset is a tuple of t points
        # We can precompute for each point which t-subsets it's in
        point_uncovered_count = np.zeros(v, dtype=np.int32)
        for p in range(v):
            # point_to_ts[p] contains indices of t-subsets containing p
            if len(point_to_ts[p]) > 0:
                point_uncovered_count[p] = np.sum(uncovered_mask[point_to_ts[p]])

        block = []
        points_used = set()
        current_uncovered = uncovered_mask.copy()

        for _ in range(k):
            # Find point with max uncovered count
            masked_counts = point_uncovered_count.copy()
            for p in points_used:
                masked_counts[p] = -1
            best_point = int(np.argmax(masked_counts))
            best_count = masked_counts[best_point]

            if best_count <= 0:
                break

            block.append(best_point)
            points_used.add(best_point)

            # Update: remove t-subsets that are now fully contained in block
            # A t-subset is fully in block if all its t points are in block
            for ts_idx in np.where(current_uncovered)[0]:
                ts = all_t_subsets[ts_idx]
                if all(p in points_used for p in ts):
                    current_uncovered[ts_idx] = False

            # Update point counts
            for p in range(v):
                if p in points_used:
                    point_uncovered_count[p] = 0
                else:
                    if len(point_to_ts[p]) > 0:
                        point_uncovered_count[p] = np.sum(current_uncovered[point_to_ts[p]])

        if len(block) < k:
            remaining = [p for p in range(v) if p not in points_used]
            rng.shuffle(remaining)
            for p in remaining:
                if len(block) >= k:
                    break
                block.append(p)

        block.sort()
        return tuple(block)

    def add_block(block):
        """Add a block to the solution and update coverage"""
        if block in selected_set:
            return False
        selected_set.add(block)
        selected_blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = ts_index[ts]
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered[idx] = False
        return True

    def prune_redundant():
        """Remove redundant blocks"""
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(selected_blocks)):
                if not time_ok():
                    break
                block = selected_blocks[i]
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = ts_index[ts]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(block, t):
                        idx = ts_index[ts]
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            uncovered[idx] = True
                    selected_blocks.pop(i)
                    selected_set.discard(block)
                    changed = True
                    break
        return selected_blocks

    def local_search_swap():
        """Try to swap blocks to reduce count while maintaining coverage"""
        if not time_ok():
            return
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(selected_blocks)):
                if not time_ok():
                    break
                block_i = selected_blocks[i]
                # Find t-subsets uniquely covered by block_i
                unique_ts = []
                for ts in itertools.combinations(block_i, t):
                    idx = ts_index[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)

                if not unique_ts:
                    # Redundant, remove
                    for ts in itertools.combinations(block_i, t):
                        idx = ts_index[ts]
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            uncovered[idx] = True
                    selected_blocks.pop(i)
                    selected_set.discard(block_i)
                    improved = True
                    break

                # Find a candidate block that covers all unique_ts
                # and provides maximum additional coverage
                best_candidate = None
                best_new_cover = -1

                # Generate candidate blocks using density approach targeting unique_ts
                unique_ts_set = set(unique_ts)
                for attempt in range(5):
                    cand = fast_density_block(np.array([idx in unique_ts_set for idx in range(num_t_subsets)]), random)
                    if cand is None or cand in selected_set:
                        continue

                    # Check if cand covers all unique_ts
                    cand_ts_set = set()
                    for ts in itertools.combinations(cand, t):
                        cand_ts_set.add(ts_index[ts])
                    if not all(ts_idx in cand_ts_set for ts_idx in unique_ts):
                        continue

                    # Count new coverage
                    new_cover = 0
                    for ts_idx in cand_ts_set:
                        if cover_count[ts_idx] == 0:
                            new_cover += 1

                    if new_cover > best_new_cover:
                        best_new_cover = new_cover
                        best_candidate = cand

                if best_candidate is not None:
                    # Perform swap
                    for ts in itertools.combinations(block_i, t):
                        idx = ts_index[ts]
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            uncovered[idx] = True

                    selected_blocks[i] = best_candidate
                    selected_set.discard(block_i)
                    selected_set.add(best_candidate)

                    for ts in itertools.combinations(best_candidate, t):
                        idx = ts_index[ts]
                        cover_count[idx] += 1
                        if cover_count[idx] == 1:
                            uncovered[idx] = False

                    improved = True
                    break
        return selected_blocks

    # Main construction loop
    rng = random.Random(42)

    # Phase 1: Density-driven construction
    while time_ok():
        if not np.any(uncovered):
            break

        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break

        # Generate a block targeting uncovered t-subsets
        block = fast_density_block(uncovered, rng)
        if block is None:
            break

        if add_block(block):
            pass
        else:
            # If block already exists, try a different one
            # Add some randomness
            block = list(block)
            for _ in range(10):
                if len(block) > 0:
                    block.pop(rng.randint(0, len(block)-1))
                block.append(rng.randint(0, v-1))
                block = sorted(set(block))
                if len(block) == k:
                    break
            if len(block) == k:
                block = tuple(block)
                add_block(block)

    # Phase 2: Prune redundant blocks
    prune_redundant()

    # Phase 3: Local search for swaps
    local_search_swap()

    # Phase 4: More rounds of density construction + prune + local search
    for round_num in range(20):
        if not time_ok():
            break

        # Check if fully covered
        if not np.any(uncovered):
            break

        # Generate additional blocks
        for _ in range(5):
            if not time_ok():
                break
            if not np.any(uncovered):
                break
            block = fast_density_block(uncovered, rng)
            if block is not None:
                add_block(block)

        prune_redundant()
        if time_ok():
            local_search_swap()

    # Final prune
    prune_redundant()

    # Convert to list of lists
    return [list(block) for block in selected_blocks]
