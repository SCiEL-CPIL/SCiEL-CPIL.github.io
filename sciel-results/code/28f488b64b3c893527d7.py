import time
import random
import itertools
import heapq
import numpy as np
from collections import defaultdict

_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_subsets = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each block, list of t-subsets it covers
    block_subsets = []
    # For each t-subset, list of blocks that cover it
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    
    for i, b in enumerate(all_blocks):
        subs = []
        for s in itertools.combinations(b, t):
            s_idx = subset_to_idx[s]
            subs.append(s_idx)
            blocks_covering_subset[s_idx].append(i)
        block_subsets.append(subs)
        
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng, seed_offset=0):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    # counts[s] = number of selected blocks covering subset s
    counts = np.zeros(n_subsets, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Heap entries: (-gain, block_idx)
    # We need to lazily update gains or use a simpler approach for speed.
    # Given n_blocks ~ 50388 and n_subsets ~ 3876, full re-scan is fast enough if optimized.
    # However, heap is better. Let's use a heap with lazy deletion or just re-scan if fast.
    # Re-scan: 50k items * 150 steps = 7.5M ops. In Python, this might be ~0.5s - 1s.
    # Heap: Push/Pop is log N. But updating gains for all blocks covering a subset is expensive.
    # Let's stick to a fast greedy with numpy where possible or simple loops.
    
    # Initial gain is len(block_subsets[i])
    # We will use a simple loop but break early if time is tight.
    
    while time.time() < deadline:
        best_block = -1
        best_gain = 0
        
        # Find block with max new coverage
        # Optimization: Only consider blocks that have some new coverage.
        # We can maintain a list of candidate blocks.
        
        # Simple approach: Iterate through all blocks.
        # To speed up, we can skip blocks in selected_set.
        
        for i in range(n_blocks):
            if i in selected_set:
                continue
            
            # Calculate gain: number of subsets s in block i where counts[s] == 0
            # This is O(len(block_subsets[i])) = C(7,4) = 35 per block.
            # Total per step: 50388 * 35 ~ 1.7M operations.
            # 150 steps * 1.7M = 255M ops. This is too slow in pure Python (likely > 10s).
            
            # We need a faster greedy.
            # Use a priority queue of gains.
            # Initially, gain[i] = 35 for all i.
            # When we pick block b, we update counts.
            # For any subset s in b that was 0, it becomes 1.
            # For any block b' covering s, its gain decreases by 1.
            
            pass # Placeholder for logic, will implement efficient version below
    
    # Efficient Greedy Implementation
    # Precompute block_subsets as lists of indices
    # We need a way to quickly find max gain.
    # Let's use a heap.
    # Heap entries: (-gain, block_id, version)
    # We maintain current_gain[block_id]
    
    current_gain = [len(block_subsets[i]) for i in range(n_blocks)]
    heap = [(-current_gain[i], i, 0) for i in range(n_blocks)]
    heapq.heapify(heap)
    versions = [0] * n_blocks
    
    # We need to update gains for blocks covering a subset when that subset becomes covered.
    # This is the expensive part.
    
    # Alternative: Randomized Greedy with limited scan.
    # Or, since v=19 is small, maybe a different strategy.
    
    # Let's try a simpler, faster greedy that might not be optimal but is fast.
    # Randomize the order of blocks? No, greedy is good.
    
    # Let's optimize the gain calculation.
    # If we use NumPy for counts, we can vectorize the gain calculation for a batch of blocks?
    # No, we need the max.
    
    # Let's stick to the heap but be smart about updates.
    # Actually, for C(19,7,4), the number of blocks is ~50k.
    # Greedy usually picks ~150-200 blocks.
    # If we just re-scan all 50k blocks every step, it's 150 * 50k * 35 checks = 262.5M checks.
    # In PyPy or fast Python, this might take 5-10 seconds.
    # We have 120 seconds. We can afford a few greedy runs.
    
    # Let's implement the re-scan version as it's simpler and less bug-prone than lazy heap updates.
    # To make it faster, we can stop early if we find a block with gain 0? No, we need max.
    
    # We will use a heuristic: If time is running out, return best so far.
    
    uncovered = np.ones(n_subsets, dtype=np.int8)
    n_uncovered = n_subsets
    
    while n_uncovered > 0 and time.time() < deadline:
        best_block = -1
        best_score = -1
        
        # We can shuffle the search order to introduce randomness in ties
        # But standard greedy is deterministic.
        
        for i in range(n_blocks):
            if i in selected_set:
                continue
            
            gain = 0
            # Check how many uncovered subsets this block covers
            for s_idx in block_subsets[i]:
                if uncovered[s_idx] == 1:
                    gain += 1
                    if gain > best_score:
                        # Early termination for this block's gain calc? No, we need full gain to compare.
                        # But if gain > best_score, we can keep going for this block, but break outer? No.
                        pass
            
            if gain > best_score:
                best_score = gain
                best_block = i
                
        if best_block == -1 or best_score == 0:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        for s_idx in block_subsets[best_block]:
            if uncovered[s_idx] == 1:
                uncovered[s_idx] = 0
                n_uncovered -= 1
                
    return selected, selected_set

def _local_search(selected, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    # Initialize counts
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b in selected:
        for s_idx in block_subsets[b]:
            counts[s_idx] += 1
            
    # 1. Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate in reverse to allow popping from list
        for i in range(len(selected) - 1, -1, -1):
            b = selected[i]
            if b not in selected_set:
                continue
            # Check if any subset is only covered by this block
            redundant = True
            for s_idx in block_subsets[b]:
                if counts[s_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for s_idx in block_subsets[b]:
                    counts[s_idx] -= 1
                selected_set.remove(b)
                selected.pop(i)
                changed = True
                
    # 2. Simulated Annealing / Local Search for swaps
    # We want to find a block B_in that can replace B_out such that:
    # 1. B_in covers all critical subsets of B_out (subsets with count 1 in B_out)
    # 2. Ideally, B_in covers other uncovered or low-count subsets to allow further reduction later.
    # Since we are minimizing size, we look for "neutral" swaps that might enable future reductions.
    # Or, we can try to find a B_in that covers the criticals of B_out AND has high "potential" gain.
    
    # A stronger move: 1-for-1 swap.
    # If we find a swap that results in the same size, we might get stuck.
    # To improve, we can allow temporary increase in size (adding a block) if it leads to a reduction of 2+ later.
    # This is complex. Let's stick to 1-for-1 swaps that strictly maintain coverage.
    
    temperature = 10.0
    cooling_rate = 0.95
    min_temperature = 0.1
    
    current_size = len(selected)
    
    while time.time() < deadline - 5:
        if temperature < min_temperature:
            break
            
        # Pick a random block to try to remove/swap
        if not selected:
            break
            
        idx_out = rng.randint(0, len(selected) - 1)
        b_out = selected[idx_out]
        
        # Find critical subsets for b_out
        crit_subs = []
        for s_idx in block_subsets[b_out]:
            if counts[s_idx] == 1:
                crit_subs.append(s_idx)
                
        if not crit_subs:
            # This block is redundant, remove it (handled in step 1, but maybe became redundant due to swaps)
            for s_idx in block_subsets[b_out]:
                counts[s_idx] -= 1
            selected_set.remove(b_out)
            selected.pop(idx_out)
            current_size -= 1
            continue
            
        # Find candidate blocks that cover all crit_subs
        # Intersect the lists of blocks covering each crit_sub
        candidates = set(blocks_covering_subset[crit_subs[0]])
        for s_idx in crit_subs[1:]:
            candidates &= set(blocks_covering_subset[s_idx])
            if not candidates:
                break
                
        candidates.discard(b_out)
        candidates -= selected_set
        
        if not candidates:
            # No valid swap found for this block. Try another.
            continue
            
        # Score candidates
        # We want a candidate that covers many *other* subsets to increase flexibility.
        # Specifically, subsets that are currently covered exactly once (making them fragile) or zero (impossible in valid cover).
        # Actually, we want to cover subsets that are "hard" to cover?
        # Let's score by: number of subsets covered by candidate where current count is 1.
        # This suggests the candidate is "valuable" for those subsets.
        # Wait, if we swap in a block that covers many count-1 subsets, it increases their count to 2.
        # This might make the block we removed (b_out) easier to remove? No, b_out is being removed.
        # The goal of a neutral swap is to change the structure so that *other* blocks become redundant.
        
        best_candidate = -1
        best_score = -1
        
        # Randomly sample candidates if list is too large?
        # Number of candidates is usually small.
        for b_in in candidates:
            # Calculate score
            score = 0
            for s_idx in block_subsets[b_in]:
                c = counts[s_idx]
                if c == 0:
                    score += 100 # Should not happen if b_in covers crit_subs of a valid cover, but safe
                elif c == 1:
                    score += 5 # Increases robustness of this subset
                elif c == 2:
                    score += 1
            
            # Add some randomness
            score += rng.random()
            
            if score > best_score:
                best_score = score
                best_candidate = b_in
                
        if best_candidate == -1:
            continue
            
        # Evaluate the swap
        # Remove b_out, Add b_in
        # Check if this swap allows any immediate removal?
        # We'll just perform the swap and let the next loop's redundancy check handle it.
        
        # Simulate swap
        new_counts = counts.copy()
        for s_idx in block_subsets[b_out]:
            new_counts[s_idx] -= 1
        for s_idx in block_subsets[best_candidate]:
            new_counts[s_idx] += 1
            
        # Check validity: all counts must be >= 1
        # This is guaranteed if b_in covers all crit_subs of b_out.
        # Because for any s in b_out, count[s] >= 1.
        # If count[s] == 1, s is in crit_subs, so b_in covers s, new_count[s] = 1 - 1 + 1 = 1.
        # If count[s] > 1, new_count[s] >= 1.
        # For s not in b_out, count[s] >= 1, and b_in adds to it, so new_count[s] >= 1.
        
        # Acceptance criteria
        # Since size is same, we accept if it improves some heuristic "potential".
        # Or accept with probability based on "improvement" in score?
        # For now, always accept if it's a valid swap, to explore space.
        # This is effectively Random Walk on the space of covers of size N.
        
        # Perform swap
        for s_idx in block_subsets[b_out]:
            counts[s_idx] -= 1
        selected_set.remove(b_out)
        selected[idx_out] = best_candidate
        selected_set.add(best_candidate)
        
        for s_idx in block_subsets[best_candidate]:
            counts[s_idx] += 1
            
        # Try to remove redundancies immediately
        # (Optimization: do this in the main loop)
        
        temperature *= cooling_rate
        
    return selected, selected_set, counts

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0 # Safety margin
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    best_selected = None
    best_size = float('inf')
    
    # Multi-start
    num_starts = 5
    time_per_start = (deadline - t0) / num_starts
    
    for i in range(num_starts):
        start_time = time.time()
        local_deadline = start_time + time_per_start - 2.0 # Leave time for polish
        
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(123 + i)
        
        # Greedy
        selected, selected_set = _greedy_construct(v, k, t, local_deadline, rng, i)
        
        if not selected:
            continue
            
        # Local Search
        selected, selected_set, counts = _local_search(
            selected, selected_set, block_subsets, blocks_covering_subset, 
            n_subsets, local_deadline, rng
        )
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            
    # If no result found (shouldn't happen), fallback
    if best_selected is None:
        selected, selected_set = _greedy_construct(v, k, t, deadline - 1, random.Random(0), 0)
        best_selected = selected
    
    # Final Polish on the best solution
    if time.time() < deadline - 2:
        # Re-run local search on the best found so far with remaining time
        selected_set = set(best_selected)
        best_selected, selected_set, counts = _local_search(
            best_selected, selected_set, block_subsets, blocks_covering_subset,
            n_subsets, deadline - 1, random.Random(999)
        )
        
    return [list(all_blocks[i]) for i in best_selected]
