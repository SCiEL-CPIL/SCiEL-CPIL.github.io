"""Covering design C(v,k,t) via WalkSAT-style stochastic local search with NumPy vectorization."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        block_to_tsubs.append(indices)

    # Reverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, indices in enumerate(block_to_tsubs):
        for idx in indices:
            tsub_to_blocks[idx].append(bi)

    def greedy_build(seed):
        """Fast randomized greedy covering."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks_used = []
        used_block_indices = set()

        while True:
            if uncovered.sum() == 0:
                break
            if time.time() > deadline - 20:
                break

            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break

            best_block_idx = -1
            best_gain = -1
            best_block_tsubs = []

            # Sample some uncovered t-subsets, pick best block among their candidates
            sample_size = min(50, len(uncovered_indices))
            sample_idx = rng.sample(uncovered_indices.tolist(), sample_size)

            for ts_idx in sample_idx:
                covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
                if not covering_blocks:
                    continue
                rng.shuffle(covering_blocks)
                for bi in covering_blocks[:10]:
                    block_tsubs = block_to_tsubs[bi]
                    gain = sum(1 for idx in block_tsubs if uncovered[idx] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs

            if best_block_idx == -1:
                # Fallback
                ts_idx = uncovered_indices[0]
                for bi in tsub_to_blocks[ts_idx]:
                    if bi in used_block_indices:
                        continue
                    block_tsubs = block_to_tsubs[bi]
                    gain = sum(1 for idx in block_tsubs if uncovered[idx] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs

            if best_block_idx == -1:
                break

            used_block_indices.add(best_block_idx)
            blocks_used.append(all_blocks[best_block_idx])
            for idx in best_block_tsubs:
                uncovered[idx] = 0

        return blocks_used

    def reduce_blocks(blocks, max_passes=3):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks

        blocks_list = list(blocks)

        for _ in range(max_passes):
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs_list = []
            for block in blocks_list:
                indices = block_to_tsubs[all_blocks.index(tuple(block))] if tuple(block) in [tuple(b) for b in all_blocks] else []
                # Faster: compute directly
                indices = []
                for ts in itertools.combinations(tuple(block), t):
                    idx = tsub_index.get(ts)
                    if idx is not None:
                        indices.append(idx)
                block_tsubs_list.append(indices)
                for idx in indices:
                    cov_count[idx] += 1

            improved = True
            while improved:
                improved = False
                order = list(range(len(blocks_list)))
                random.shuffle(order)
                for i in order:
                    if i >= len(blocks_list):
                        continue
                    redundant = True
                    for idx in block_tsubs_list[i]:
                        if cov_count[idx] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for idx in block_tsubs_list[i]:
                            cov_count[idx] -= 1
                        blocks_list.pop(i)
                        block_tsubs_list.pop(i)
                        improved = True

            if not improved:
                break

        return blocks_list

    def walksat_local_search(initial_blocks, rng, max_iters=2000):
        """
        WalkSAT-style: maintain a set of blocks. Moves:
          1. Remove a random block; if still valid, keep removal.
          2. If invalid, add a block that covers the most newly-uncovered t-subsets (biased random).
          3. Occasionally swap: remove one, add a different one.
        Goal: minimize number of blocks while maintaining coverage.
        """
        if not initial_blocks:
            return initial_blocks

        current = list(initial_blocks)
        n = len(current)

        # Build coverage structure
        # For speed, use block index mapping
        block_set = set(tuple(b) for b in current)
        # Map block tuple to its t-sub indices
        block_tsub_cache = {}
        for b in current:
            bt = tuple(b)
            if bt not in block_tsub_cache:
                indices = []
                for ts in itertools.combinations(bt, t):
                    idx = tsub_index.get(ts)
                    if idx is not None:
                        indices.append(idx)
                block_tsub_cache[bt] = indices

        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in current:
            for idx in block_tsub_cache[tuple(b)]:
                cov_count[idx] += 1

        best_size = n
        best_blocks = list(current)
        no_improve_count = 0

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            move = rng.random()

            if move < 0.4:
                # Try to remove a random block
                if n <= 1:
                    break
                i = rng.randrange(n)
                saved_block = current[i]
                saved_tsubs = block_tsub_cache[tuple(saved_block)]

                # Remove
                current.pop(i)
                for idx in saved_tsubs:
                    cov_count[idx] -= 1

                if cov_count.min() >= 1:
                    # Valid removal!
                    n -= 1
                    if n < best_size:
                        best_size = n
                        best_blocks = list(current)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Restore
                    current.insert(i, saved_block)
                    for idx in saved_tsubs:
                        cov_count[idx] += 1
                    # Try to add a replacement block that helps
                    uncovered_idx = np.where(cov_count == 1)[0]  # t-subsets that would be uncovered if we removed
                    if len(uncovered_idx) > 0:
                        # Pick a random uncovered t-subset, find a block covering it that's not current
                        ts_idx = uncovered_idx[rng.randrange(len(uncovered_idx))]
                        candidates = [bi for bi in tsub_to_blocks[ts_idx] if tuple(all_blocks[bi]) not in block_set]
                        if candidates:
                            # Pick one with best gain
                            best_cand = -1
                            best_gain = -1
                            for bi in candidates[:20]:
                                cand_tsubs = block_to_tsubs[bi]
                                gain = sum(1 for idx in cand_tsubs if cov_count[idx] <= 1)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_cand = bi
                            if best_cand != -1:
                                new_block = all_blocks[best_cand]
                                current.append(new_block)
                                block_set.add(tuple(new_block))
                                new_tsubs = block_to_tsubs[best_cand]
                                for idx in new_tsubs:
                                    cov_count[idx] += 1
                                n += 1
                                if n < best_size:
                                    best_size = n
                                    best_blocks = list(current)
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                            else:
                                no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

            elif move < 0.7:
                # Swap: remove one block, add a different one
                if n <= 1:
                    no_improve_count += 1
                    continue
                i = rng.randrange(n)
                saved_block = current[i]
                saved_tsubs = block_tsub_cache[tuple(saved_block)]

                # Remove
                current.pop(i)
                for idx in saved_tsubs:
                    cov_count[idx] -= 1

                # Find uncovered t-subsets
                uncov = np.where(cov_count == 0)[0]
                if len(uncov) > 0:
                    # Find a block to add that covers the most uncovered
                    best_cand = -1
                    best_gain = -1
                    # Sample from candidates of a few uncovered t-subsets
                    sample_uncov = uncov[:min(10, len(uncov))]
                    candidate_set = set()
                    for ts_idx in sample_uncov:
                        for bi in tsub_to_blocks[ts_idx]:
                            if tuple(all_blocks[bi]) not in block_set:
                                candidate_set.add(bi)
                    for bi in list(candidate_set)[:30]:
                        cand_tsubs = block_to_tsubs[bi]
                        gain = sum(1 for idx in cand_tsubs if cov_count[idx] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = bi

                    if best_cand != -1:
                        new_block = all_blocks[best_cand]
                        current.append(new_block)
                        block_set.add(tuple(new_block))
                        new_tsubs = block_to_tsubs[best_cand]
                        for idx in new_tsubs:
                            cov_count[idx] += 1
                        # n stays same
                        no_improve_count += 1
                    else:
                        # Restore
                        current.insert(i, saved_block)
                        for idx in saved_tsubs:
                            cov_count[idx] += 1
                        no_improve_count += 1
                else:
                    # No uncovered, just restore (swap didn't help)
                    current.insert(i, saved_block)
                    for idx in saved_tsubs:
                        cov_count[idx] += 1
                    no_improve_count += 1

            else:
                # Random restart within current solution: shuffle and try aggressive reduction
                if n <= 1:
                    no_improve_count += 1
                    continue
                # Try removing multiple random blocks in sequence
                removed = []
                for _ in range(min(3, n)):
                    if n <= len(removed):
                        break
                    i = rng.randrange(len(current))
                    saved_block = current[i]
                    saved_tsubs = block_tsub_cache.get(tuple(saved_block), [])
                    current.pop(i)
                    for idx in saved_tsubs:
                        cov_count[idx] -= 1
                    removed.append((saved_block, saved_tsubs))

                if cov_count.min() >= 1:
                    n = len(current)
                    if n < best_size:
                        best_size = n
                        best_blocks = list(current)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Restore all
                    for saved_block, saved_tsubs in removed:
                        current.append(saved_block)
                        for idx in saved_tsubs:
                            cov_count[idx] += 1
                    no_improve_count += 1

            if no_improve_count > 500:
                break

        return best_blocks

    # Main search
    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple greedy constructions
    for seed in range(30):
        if time.time() > deadline - 30:
            break
        blocks = greedy_build(seed)
        reduced = reduce_blocks(blocks, max_passes=2)
        if len(reduced) < best_count:
            best_blocks = reduced
            best_count = len(reduced)

    # Phase 2: WalkSAT local search from best and other solutions
    if best_blocks is not None:
        rng = random.Random(42)
        # Run WalkSAT from best solution
        if time.time() < deadline - 10:
            result = walksat_local_search(best_blocks, rng, max_iters=3000)
            if len(result) < best_count:
                best_blocks = result
                best_count = len(result)

        # Run WalkSAT from a few other greedy solutions
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            blocks = greedy_build(100 + seed)
            reduced = reduce_blocks(blocks, max_passes=1)
            if time.time() > deadline - 10:
                break
            result = walksat_local_search(reduced, rng, max_iters=2000)
            if len(result) < best_count:
                best_blocks = result
                best_count = len(result)

    # Final reduction
    if best_blocks is not None and time.time() < deadline - 2:
        best_blocks = reduce_blocks(best_blocks, max_passes=5)

    return best_blocks if best_blocks else greedy_build(0)
