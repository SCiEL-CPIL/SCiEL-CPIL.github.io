import itertools
import random
import time
import numpy as np
from collections import defaultdict

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings efficiently."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    block_idx = {b: i for i, b in enumerate(blocks)}
    
    # For each block, list of t-subset indices it covers
    block_to_ts = [[] for _ in range(n_blocks)]
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(n_tsubs)]
    
    # Precompute all t-subsets of each block more efficiently
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsi = tsub_idx[ts]
            block_to_ts[bi].append(tsi)
            ts_to_blocks[tsi].append(bi)
            
    # Convert to numpy arrays for faster access in loops
    block_to_ts_arr = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    # Precompute the count of blocks covering each t-subset (for heuristic ordering)
    ts_cover_count = np.array([len(lst) for lst in ts_to_blocks], dtype=np.int32)
    
    return tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count

def _run_greedy(v, k, t, seed, precomp, deadline):
    """Vectorized randomized greedy set cover with better gain updates."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    rng = random.Random(seed)
    
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.full(n_blocks, C(k, t), dtype=np.int32)
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected = []
    
    # Track which blocks are still available
    available = np.ones(n_blocks, dtype=bool)
    
    while np.any(uncovered):
        if time.time() > deadline - 5.0:
            return None # Timeout, return None to indicate failure to complete
        
        # Find block with max gain among available blocks
        avail_gain = np.where(available, gain, -1)
        max_gain = np.max(avail_gain)
        if max_gain <= 0:
            break
            
        # Random tie-breaking among available blocks with max gain
        candidates = np.where(avail_gain == max_gain)[0]
        best_bi = candidates[rng.randrange(len(candidates))]
        
        selected.append(best_bi)
        available[best_bi] = False
        
        # Get t-subsets covered by this block
        covered_ts = block_to_ts_arr[best_bi]
        
        # Identify which of these were actually uncovered
        was_uncovered_mask = uncovered[covered_ts]
        newly_covered_ts = covered_ts[was_uncovered_mask]
        
        if len(newly_covered_ts) == 0:
            continue
            
        # Update uncovered status
        uncovered[newly_covered_ts] = False
        
        # Update gains: for each newly covered t-subset, decrement gain of all blocks covering it
        # Use vectorized approach with bincount
        all_block_indices = np.concatenate([ts_to_blocks_arr[tsi] for tsi in newly_covered_ts]) if len(newly_covered_ts) > 0 else np.array([], dtype=np.int32)
        
        if len(all_block_indices) > 0:
            counts = np.bincount(all_block_indices, minlength=n_blocks)
            gain[available] -= counts[available]
            # Ensure gains don't go negative
            gain = np.maximum(gain, 0)
            gain[~available] = -1
        
    return selected

def _local_search(v, k, t, block_indices, precomp, deadline):
    """Local search: redundancy removal and 1-swaps with more aggressive search."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    rng = random.Random(42)
    
    # Work with a list of block indices
    sol = list(block_indices)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Remove redundant blocks (multiple passes)
        # Shuffle to avoid bias
        indices = list(range(len(sol)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2.0:
                return sol
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            if np.all(cov[ts_idxs] > 1):
                # Redundant
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break # Restart loop to re-evaluate redundancies
                
        if improved:
            continue
            
        # Phase 2: Try 1-swaps - be more aggressive
        # Try to replace a block with another that covers its critical t-subsets
        num_swaps_tried = 0
        max_swaps = 50  # Increased from 20
        
        while num_swaps_tried < max_swaps and time.time() < deadline - 2.0:
            num_swaps_tried += 1
            if len(sol) == 0:
                break
                
            i = rng.randrange(len(sol))
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            
            # Find critical t-subsets (covered only by bi)
            critical_mask = cov[ts_idxs] == 1
            critical_ts = ts_idxs[critical_mask]
            
            if len(critical_ts) == 0:
                # Should have been removed in Phase 1
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break
                
            # Find a replacement block that covers all critical_ts
            # Pick the critical t-subset with the fewest covering blocks to minimize search space
            crit_sorted = sorted(critical_ts, key=lambda x: len(ts_to_blocks_arr[x]))
            first_crit = crit_sorted[0]
            
            found_replacement = False
            # Iterate through blocks covering first_crit
            # Randomize order to escape local minima
            candidates = list(ts_to_blocks_arr[first_crit])
            rng.shuffle(candidates)
            
            for bi2 in candidates:
                if bi2 == bi:
                    continue
                # Check if bi2 covers all critical_ts
                ts_idxs_bi2 = block_to_ts_arr[bi2]
                # Check if critical_ts is a subset of ts_idxs_bi2
                # Use set intersection for clarity and speed with small sets
                if set(critical_ts).issubset(set(ts_idxs_bi2)):
                    # Valid replacement
                    # Update coverage
                    cov[ts_idxs] -= 1
                    cov[ts_idxs_bi2] += 1
                    sol[i] = bi2
                    improved = True
                    found_replacement = True
                    break
                    
            if found_replacement:
                break # Restart loop to check for new redundancies
                
    return sol

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

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy restarts
    # Allocate ~70% of time to greedy, ~30% to local search
    greedy_deadline = start_time + (deadline - start_time) * 0.7
    num_restarts = 0
    while time.time() < greedy_deadline:
        seed = num_restarts
        blocks = _run_greedy(v, k, t, seed, precomp, deadline)
        num_restarts += 1
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
    if best_blocks is None:
        # Fallback: try one more greedy with full deadline
        blocks = _run_greedy(v, k, t, 0, precomp, deadline)
        if blocks is not None:
            best_blocks = blocks
        else:
            # Extreme fallback: just pick blocks until covered (slow, but safe)
            uncovered = set(itertools.combinations(range(v), t))
            best_blocks = []
            while uncovered and time.time() < deadline - 5.0:
                target = next(iter(uncovered))
                best_block = None
                best_gain = -1
                remaining = [x for x in range(v) if x not in target]
                need = k - t
                for comb in itertools.combinations(remaining, need):
                    block = tuple(sorted(target + comb))
                    gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                if best_block is None:
                    break
                best_blocks.append(best_block)
                uncovered.difference_update(itertools.combinations(best_block, t))
            return [list(b) for b in best_blocks]

    # Local search on the best solution with remaining time
    if time.time() < deadline - 3.0:
        best_blocks = _local_search(v, k, t, best_blocks, precomp, deadline)
        
    # Convert indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, ts_cover_count = precomp
    final_blocks = []
    seen = set()
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
