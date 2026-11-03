"""
Covering design solver using dynamic greedy with heap-based priority updates.
Strategy: maintain a max-heap of candidate blocks keyed by number of uncovered t-subsets
they would cover. After each block selection, efficiently update only affected candidates.
"""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import heapq
from collections import defaultdict

_DEADLINE = 140.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    rng = random.Random(12345)

    # All t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Track coverage: which t-subsets are uncovered
    uncovered = [True] * n_ts
    uncovered_count = n_ts

    # For each point, store which t-subsets it appears in
    # point_to_ts[point] = list of indices of t-subsets containing that point
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_ts):
        for p in ts:
            point_to_ts[p].append(i)

    # For each (k-1)-subset, store which points can extend it to a k-subset
    # and precompute the t-subsets that would be covered
    # Actually, let's use a different approach:
    # A block is a k-subset. The t-subsets it covers are all t-subsets within it.
    # Number of t-subsets in a k-subset = C(k, t)
    #
    # Dynamic greedy:
    # 1. Start with no blocks.
    # 2. At each step, find the k-subset that covers the most uncovered t-subsets.
    # 3. Add it, mark its t-subsets as covered.
    # 4. Repeat until all covered or time runs out.
    #
    # To speed up step 2, we can use the fact that a k-subset's coverage is determined
    # by which of its C(k,t) t-subsets are still uncovered.
    #
    # Efficient approach: 
    # - For each point, maintain a count of how many uncovered t-subsets contain it.
    # - When we select a block, for each t-subset in the block, we decrement counts
    #   for all points in that t-subset.
    # - The "score" of a potential new block is the sum of uncovered t-subsets in it.
    #
    # But enumerating all C(v,k) blocks is infeasible for v=19, k=7 (C(19,7)=50388).
    # Actually C(19,7) = 50388, which is manageable! Let's precompute all blocks.
    #
    # C(19,7) = 50388 blocks, each covering C(7,4) = 35 t-subsets.
    # Total t-subsets: C(19,4) = 3876.
    #
    # So we can precompute for each block which t-subsets it covers, and use
    # a heap-based approach where we track the current "gain" of each block.

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    def _greedy_cover(uncovered, uncovered_count):
        """Run greedy set cover, return list of selected block indices."""
        # Current uncovered status
        unc = [True] * n_ts
        unc_count = uncovered_count
        for s_idx in range(n_ts):
            unc[s_idx] = uncovered[s_idx]

        # For each block, track how many of its t-subsets are still uncovered
        block_gain = [0] * n_blocks
        for b_idx in range(n_blocks):
            cnt = 0
            for s_idx in block_ts[b_idx]:
                if unc[s_idx]:
                    cnt += 1
            block_gain[b_idx] = cnt

        # Heap with lazy deletion
        block_version = [0] * n_blocks
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks) if block_gain[i] > 0]
        heapq.heapify(heap)

        selected = []
        in_block = [False] * n_blocks

        while unc_count > 0 and _remaining() > 1.0:
            # Pop from heap until we find a valid entry
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update: for each t-subset in this block, mark as covered and update gains
            for s_idx in block_ts[b_idx]:
                if not unc[s_idx]:
                    continue
                unc[s_idx] = False
                unc_count -= 1

                # For each block that contains this t-subset, decrement its gain
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        return selected

    # Initial greedy cover
    initial_uncovered = [True] * n_ts
    selected_indices = _greedy_cover(initial_uncovered, n_ts)
    result = [list(all_blocks[b_idx]) for b_idx in selected_indices]

    # Local search: try to remove redundant blocks
    if _remaining() > 5.0 and result:
        # Build coverage counts
        coverage = [0] * n_ts
        for blk in result:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Try to remove blocks that are redundant
        changed = True
        while changed and _remaining() > 2.0:
            changed = False
            for i in range(len(result) - 1, -1, -1):
                if _remaining() < 1.0:
                    break
                blk = result[i]
                redundant = True
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        coverage[ts_index[ts]] -= 1
                    result.pop(i)
                    changed = True
                    break

    # Swap-based local search: try to replace one block with another to reduce total
    # This is more complex; for now, do multiple restarts with different seeds
    if _remaining() > 30.0:
        best_result = result[:]
        best_len = len(result)
        
        # Try a few restarts with different random seeds
        for restart in range(5):
            if _remaining() < 15.0:
                break
            
            # Randomly perturb: remove some blocks and re-cover
            perturbation_size = max(1, len(best_result) // 10)
            perturbed = best_result[:]
            
            # Remove random blocks
            indices_to_remove = rng.sample(range(len(perturbed)), min(perturbation_size, len(perturbed)))
            for idx in sorted(indices_to_remove, reverse=True):
                perturbed.pop(idx)
            
            # Check what's uncovered
            cov = [0] * n_ts
            for blk in perturbed:
                for ts in itertools.combinations(blk, t):
                    cov[ts_index[ts]] += 1
            
            uncovered_now = [cov[i] == 0 for i in range(n_ts)]
            unc_count = sum(uncovered_now)
            
            if unc_count == 0:
                # Still valid, but we removed blocks
                if len(perturbed) < best_len:
                    best_result = perturbed
                    best_len = len(perturbed)
                continue
            
            # Re-cover the uncovered parts
            additional = _greedy_cover(uncovered_now, unc_count)
            perturbed.extend(all_blocks[b_idx] for b_idx in additional)
            
            # Remove duplicates
            seen = set()
            unique = []
            for blk in perturbed:
                key = tuple(sorted(blk))
                if key not in seen:
                    seen.add(key)
                    unique.append(list(blk))
            perturbed = unique
            
            # Check validity
            cov = [0] * n_ts
            for blk in perturbed:
                for ts in itertools.combinations(blk, t):
                    cov[ts_index[ts]] += 1
            
            if all(c > 0 for c in cov) and len(perturbed) < best_len:
                best_result = perturbed
                best_len = len(perturbed)
        
        result = best_result

    # Final verification
    final_cov = [0] * n_ts
    for blk in result:
        for ts in itertools.combinations(blk, t):
            final_cov[ts_index[ts]] += 1
    if not all(c > 0 for c in final_cov):
        # Fallback: add greedy blocks for uncovered subsets
        unc_set = set(i for i in range(n_ts) if final_cov[i] == 0)
        while unc_set and _remaining() > 0.5:
            # Find block covering most uncovered
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = 0
                for s_idx in block_ts[b_idx]:
                    if s_idx in unc_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            blk = list(all_blocks[best_b])
            result.append(blk)
            for s_idx in block_ts[best_b]:
                unc_set.discard(s_idx)

    return result
# EVOLVE-BLOCK-END
