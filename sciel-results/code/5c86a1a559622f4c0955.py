"""
Covering design solver using bucket-based dynamic greedy + local search with swaps.
Strategy: maintain gain counts per block in buckets for O(1) max-gain lookup.
After greedy construction, perform deletion and swap local search.
Multi-restart with different seeds to escape local optima.
"""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict, deque

_DEADLINE = 143.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _greedy_construct(v, k, t, all_blocks, block_ts, ts_to_blocks, n_ts, n_blocks, seed):
    """Greedy set cover with bucket-based priority. Returns list of block indices."""
    rng = random.Random(seed)
    
    # Current uncovered status
    uncovered = [True] * n_ts
    uncovered_count = n_ts
    
    # For each block, track how many of its t-subsets are still uncovered
    block_gain = [len(ts_list) for ts_list in block_ts]
    
    # Bucket structure: buckets[gain] = set of block indices with that gain
    max_gain = max(block_gain)
    buckets = [set() for _ in range(max_gain + 1)]
    for b_idx in range(n_blocks):
        buckets[block_gain[b_idx]].add(b_idx)
    
    current_max = max_gain
    in_block = [False] * n_blocks
    selected = []
    
    while uncovered_count > 0 and _remaining() > 2.0:
        # Find the highest gain bucket with available blocks
        while current_max > 0 and not buckets[current_max]:
            current_max -= 1
        
        if current_max <= 0:
            break
        
        # Pick a block from the top bucket (random tie-breaking for diversity)
        candidates = buckets[current_max]
        b_idx = next(iter(rng.sample(list(candidates), min(len(candidates), 10))))
        candidates.discard(b_idx)
        
        selected.append(b_idx)
        in_block[b_idx] = True
        
        # Update: for each t-subset in this block, mark as covered and update gains
        for s_idx in block_ts[b_idx]:
            if not uncovered[s_idx]:
                continue
            uncovered[s_idx] = False
            uncovered_count -= 1
            
            # For each block that contains this t-subset, decrement its gain
            for other_b in ts_to_blocks[s_idx]:
                if in_block[other_b]:
                    continue
                old_gain = block_gain[other_b]
                new_gain = old_gain - 1
                block_gain[other_b] = new_gain
                buckets[old_gain].discard(other_b)
                if new_gain > 0:
                    buckets[new_gain].add(other_b)
        
        # Update current_max if needed
        if current_max > 0 and not buckets[current_max]:
            current_max -= 1
    
    return selected, uncovered


def _local_search_delete(v, k, t, selected_indices, all_blocks, block_ts, ts_index, n_ts):
    """Try to remove redundant blocks. Returns modified list of block indices."""
    if not selected_indices:
        return selected_indices
    
    # Build coverage counts
    coverage = [0] * n_ts
    for b_idx in selected_indices:
        for s_idx in block_ts[b_idx]:
            coverage[s_idx] += 1
    
    result = list(selected_indices)
    changed = True
    while changed and _remaining() > 3.0:
        changed = False
        for i in range(len(result) - 1, -1, -1):
            if _remaining() < 2.0:
                break
            b_idx = result[i]
            redundant = True
            for s_idx in block_ts[b_idx]:
                if coverage[s_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for s_idx in block_ts[b_idx]:
                    coverage[s_idx] -= 1
                result.pop(i)
                changed = True
                break
    
    return result


def _local_search_swap(v, k, t, selected_indices, all_blocks, block_ts, ts_index, n_ts, ts_to_blocks, n_blocks, seed):
    """Try to replace blocks to improve coverage efficiency. Returns modified list of block indices."""
    if not selected_indices:
        return selected_indices
    
    rng = random.Random(seed)
    result = list(selected_indices)
    selected_set = set(result)
    
    # Build coverage counts
    coverage = [0] * n_ts
    for b_idx in result:
        for s_idx in block_ts[b_idx]:
            coverage[s_idx] += 1
    
    improved = True
    iterations = 0
    while improved and _remaining() > 5.0 and iterations < 200:
        improved = False
        iterations += 1
        
        # Try removing each block and see if we can add one that covers more
        for i in range(len(result)):
            if _remaining() < 3.0:
                break
            
            b_remove = result[i]
            # Temporarily remove
            for s_idx in block_ts[b_remove]:
                coverage[s_idx] -= 1
            
            # Find uncovered t-subsets
            unc_count = sum(1 for c in coverage if c == 0)
            if unc_count == 0:
                # Just remove
                result.pop(i)
                selected_set.discard(b_remove)
                improved = True
                break
            
            # Find best block to add that covers most uncovered
            best_b = -1
            best_gain = 0
            # Sample a subset of candidate blocks for speed
            candidates = [b for b in range(n_blocks) if b not in selected_set]
            if len(candidates) > 2000:
                candidates = rng.sample(candidates, 2000)
            
            for b_idx in candidates:
                gain = 0
                for s_idx in block_ts[b_idx]:
                    if coverage[s_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            
            if best_b >= 0 and best_gain >= 1:
                # Add the new block
                result[i] = best_b
                selected_set.add(best_b)
                for s_idx in block_ts[best_b]:
                    coverage[s_idx] += 1
                improved = True
                break
            else:
                # Restore
                for s_idx in block_ts[b_remove]:
                    coverage[s_idx] += 1
    
    return result


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

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

    # Multi-restart strategy
    best_result = None
    best_size = float('inf')
    
    # Determine number of restarts based on time
    # Each greedy pass takes roughly 0.5-2s, local search takes a few seconds
    # Budget: ~140s total
    n_restarts = 5
    seeds = [12345, 67890, 11111, 22222, 33333]
    
    for restart in range(n_restarts):
        if _remaining() < 15.0:
            break
        
        seed = seeds[restart % len(seeds)]
        
        # Greedy construction
        selected_indices, uncovered = _greedy_construct(
            v, k, t, all_blocks, block_ts, ts_to_blocks, n_ts, n_blocks, seed
        )
        
        if not selected_indices:
            continue
        
        # Local search: deletion
        selected_indices = _local_search_delete(
            v, k, t, selected_indices, all_blocks, block_ts, ts_index, n_ts
        )
        
        # Local search: swap
        if _remaining() > 10.0:
            selected_indices = _local_search_swap(
                v, k, t, selected_indices, all_blocks, block_ts, ts_index, n_ts, ts_to_blocks, n_blocks, seed
            )
        
        # Verify and fix
        coverage = [0] * n_ts
        for b_idx in selected_indices:
            for s_idx in block_ts[b_idx]:
                coverage[s_idx] += 1
        
        # Fix any uncovered t-subsets
        uncovered_indices = [i for i in range(n_ts) if coverage[i] == 0]
        if uncovered_indices and _remaining() > 5.0:
            unc_set = set(uncovered_indices)
            for s_idx in uncovered_indices:
                # Find a block covering this t-subset
                for b_idx in ts_to_blocks[s_idx]:
                    if b_idx not in selected_indices:
                        selected_indices.append(b_idx)
                        for s2 in block_ts[b_idx]:
                            coverage[s2] += 1
                        unc_set.discard(s_idx)
                        break
        
        # Final deletion pass
        selected_indices = _local_search_delete(
            v, k, t, selected_indices, all_blocks, block_ts, ts_index, n_ts
        )
        
        # Verify
        final_cov = [0] * n_ts
        for b_idx in selected_indices:
            for s_idx in block_ts[b_idx]:
                final_cov[s_idx] += 1
        if all(c > 0 for c in final_cov):
            if len(selected_indices) < best_size:
                best_size = len(selected_indices)
                best_result = [list(all_blocks[b_idx]) for b_idx in selected_indices]
        else:
            # Shouldn't happen, but keep as fallback
            if best_result is None:
                best_result = [list(all_blocks[b_idx]) for b_idx in selected_indices]
    
    if best_result is None:
        # Fallback: single greedy
        selected_indices, _ = _greedy_construct(
            v, k, t, all_blocks, block_ts, ts_to_blocks, n_ts, n_blocks, 12345
        )
        best_result = [list(all_blocks[b_idx]) for b_idx in selected_indices]
    
    return best_result
# EVOLVE-BLOCK-END
