import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    # 1. Precomputation
    all_ts = list(itertools.combinations(range(v), t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    num_ts = len(all_ts)

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, store the list of t-subset indices it covers
    block_to_ts = []
    # For each t-subset, store list of block indices that cover it
    ts_to_blocks = [[] for _ in range(num_ts)]

    for b_idx, block in enumerate(all_blocks):
        bts = [ts_index[ts] for ts in itertools.combinations(block, t)]
        block_to_ts.append(bts)
        for ts_idx in bts:
            ts_to_blocks[ts_idx].append(b_idx)

    # Convert to numpy arrays for vectorized operations
    # Each block has C(k,t) = C(8,5) = 56 t-subsets
    ts_per_block = len(block_to_ts[0])
    # Pad to uniform length (they're all the same length for fixed v,k,t)
    block_ts_array = np.array(block_to_ts, dtype=np.int32)  # shape (num_blocks, ts_per_block)

    # 2. Vectorized Greedy
    def greedy_init(rng):
        uncovered = np.ones(num_ts, dtype=np.int8)
        selected = []

        while True:
            if time.time() >= deadline - 5:
                break

            # Compute gain for each block: number of uncovered t-subsets it covers
            # For each block, count how many of its t-subsets are uncovered
            # block_ts_array has shape (num_blocks, ts_per_block)
            # uncovered has shape (num_ts,)
            # gain[b] = sum(uncovered[block_ts_array[b]])
            gains = uncovered[block_ts_array].sum(axis=1)

            if gains.max() == 0:
                break

            best = np.argmax(gains)
            selected.append(best)

            # Mark covered
            uncovered[block_ts_array[best]] = 0

        # Build counts
        counts = np.zeros(num_ts, dtype=np.int32)
        for b in selected:
            counts[block_ts_array[b]] += 1

        return selected, counts

    # 3. Local Search with conductance-based removal
    def local_search(blocks, counts, rng):
        blocks = list(blocks)
        counts = counts.copy()

        improved = True
        while improved and time.time() < deadline - 3:
            improved = False

            # Phase A: Remove trivially redundant blocks (no unique t-subsets)
            n = len(blocks)
            for i in range(n):
                if time.time() >= deadline - 3:
                    break
                b = blocks[i]
                # Check if any t-subset has count == 1
                redundant = True
                for ts_idx in block_to_ts[b]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    counts[block_ts_array[b]] -= 1
                    blocks.pop(i)
                    improved = True
                    break  # restart to re-index

            if improved:
                continue

            # Phase B: Conductance-based swap
            # For each block, compute:
            #   unique_count = number of t-subsets with count==1
            #   total = ts_per_block
            #   conductance = unique_count / total (lower = more redundant mass)
            # Pick the block with lowest conductance (but > 0) and try to replace it.

            n = len(blocks)
            if n == 0:
                break

            # Compute unique counts for all blocks
            best_i = -1
            best_cond = 2.0  # max possible is 1.0, so 2.0 means not found
            for i in range(n):
                b = blocks[i]
                uc = 0
                for ts_idx in block_to_ts[b]:
                    if counts[ts_idx] == 1:
                        uc += 1
                if uc > 0:
                    cond = uc / ts_per_block
                    if cond < best_cond:
                        best_cond = cond
                        best_i = i

            if best_i == -1:
                break

            b_old = blocks[best_i]
            # Get unique t-subsets of b_old
            unique_ts = [ts_idx for ts_idx in block_to_ts[b_old] if counts[ts_idx] == 1]

            # Find candidate blocks that cover all unique_ts
            # Use the first unique_ts to filter candidates
            anchor = unique_ts[0]
            candidates = ts_to_blocks[anchor]

            best_cand = -1
            best_score = -1

            # Limit candidates for speed
            if len(candidates) > 200:
                candidates = rng.sample(candidates, 200)

            for cand in candidates:
                if cand == b_old:
                    continue
                # Check if cand covers all unique_ts
                cand_ts_set = set(block_to_ts[cand])
                valid = True
                for ts_idx in unique_ts:
                    if ts_idx not in cand_ts_set:
                        valid = False
                        break
                if not valid:
                    continue

                # Score: number of t-subsets in cand (not in b_old) that have count==1
                # This indicates potential to create new redundancies
                b_old_set = set(block_to_ts[b_old])
                score = 0
                for ts_idx in block_to_ts[cand]:
                    if ts_idx not in b_old_set and counts[ts_idx] == 1:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_cand = cand

            if best_cand != -1:
                # Perform swap
                counts[block_ts_array[b_old]] -= 1
                counts[block_ts_array[best_cand]] += 1
                blocks[best_i] = best_cand
                improved = True
                # Don't break; continue to check for more improvements in same pass
                # Actually, let's break to re-run Phase A for cascading removals
                break

        return blocks, counts

    # 4. Main Loop
    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)

    # Phase 1: Multiple greedy restarts to find good initial solutions
    # With vectorized greedy, each init is fast (~10-50ms for 335 blocks)
    # Do as many as possible in the first 30 seconds
    greedy_deadline = start_time + 30
    num_greedy = 0
    while time.time() < greedy_deadline:
        blocks, counts = greedy_init(rng)
        num_greedy += 1
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)

    # Phase 2: Local search on the best solutions
    # Sort candidates by length, take top few
    # We only kept the best, but let's also keep a few for diversity
    # Actually, let's just do local search on the best one, and if time permits,
    # do a couple more greedy restarts + local search

    if best_blocks is not None:
        # Rebuild counts for the best
        counts = np.zeros(num_ts, dtype=np.int32)
        for b in best_blocks:
            counts[block_ts_array[b]] += 1

        improved_blocks, _ = local_search(best_blocks, counts, rng)
        if len(improved_blocks) < best_count:
            best_blocks = improved_blocks
            best_count = len(improved_blocks)

    # Phase 3: If time remains, do a few more greedy + local search cycles
    while time.time() < deadline - 15:
        blocks, counts = greedy_init(rng)
        if len(blocks) < best_count:
            improved_blocks, _ = local_search(blocks, counts, rng)
            if len(improved_blocks) < best_count:
                best_blocks = improved_blocks
                best_count = len(improved_blocks)
        elif len(blocks) <= best_count + 5:
            # Close to best, try local search anyway
            improved_blocks, _ = local_search(blocks, counts, rng)
            if len(improved_blocks) < best_count:
                best_blocks = improved_blocks
                best_count = len(improved_blocks)

    if best_blocks is None:
        return []

    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
