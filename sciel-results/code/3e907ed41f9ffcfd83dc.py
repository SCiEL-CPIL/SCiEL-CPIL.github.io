import itertools
import random
import time
import sys
import heapq
from collections import defaultdict

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_masks[i] is a bitmask of triples covered by block i
    block_masks = [0] * n_blocks
    # blocks_containing[j] is a list of block indices containing triple j
    blocks_containing = [[] for _ in range(n_triples)]
    # block_triples[i] is a list of triple indices in block i (for fast iteration)
    block_triples = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        tris = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            blocks_containing[idx].append(i)
            tris.append(idx)
        block_masks[i] = mask
        block_triples[i] = tris
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, block_triples, seed):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    active_blocks = set(range(n_blocks))
    
    # Use a max-heap for gains. Store (-gain, idx) since heapq is min-heap.
    # We'll lazily invalidate stale entries.
    gain_heap = []
    gains = {}  # idx -> current gain
    
    # Initialize heap with all blocks' initial gains
    for idx in range(n_blocks):
        g = block_masks[idx].bit_count()
        gains[idx] = g
        heapq.heappush(gain_heap, (-g, idx))
    
    while uncovered:
        # Pop until we find a valid (non-stale, still active) entry
        best_block_idx = -1
        max_gain = 0
        
        while gain_heap:
            neg_gain, idx = heapq.heappop(gain_heap)
            gain = -neg_gain
            
            # Check if this entry is still valid
            if idx not in active_blocks:
                continue
            if gains.get(idx, 0) != gain:
                # Stale entry, recompute
                new_gain = (block_masks[idx] & uncovered).bit_count()
                gains[idx] = new_gain
                if new_gain > 0:
                    heapq.heappush(gain_heap, (-new_gain, idx))
                continue
            
            # Valid entry
            best_block_idx = idx
            max_gain = gain
            break
        
        if best_block_idx == -1:
            # Heap exhausted, do a full scan as fallback
            for idx in active_blocks:
                gain = (block_masks[idx] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = idx
            if best_block_idx == -1:
                break
        
        blocks_used.append(best_block_idx)
        used_mask |= block_masks[best_block_idx]
        new_uncovered = uncovered & ~block_masks[best_block_idx]
        
        # Update gains for remaining active blocks
        # Only update if it's worth it (gain changed)
        # For efficiency, we can skip updating and let lazy evaluation handle it
        # But we need to remove the selected block from active
        active_blocks.remove(best_block_idx)
        del gains[best_block_idx]
        
        uncovered = new_uncovered
        
        # Early exit if covered
        if not uncovered:
            break
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    rng = random.Random(seed)
    
    # State
    selected = set(initial_blocks)
    coverage_count = [0] * n_triples
    
    # Initialize counts
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
            
    # Helper to check if valid
    def is_valid():
        for c in coverage_count:
            if c == 0:
                return False
        return True

    if not is_valid():
        return list(initial_blocks)

    best_size = len(selected)
    best_selected = set(selected)
    
    # Precompute set version of block_triples for faster membership testing
    block_triples_set = [set(tris) for tris in block_triples]
    
    while time.time() < deadline:
        # 1. Try to remove a redundant block
        removed = False
        items = list(selected)
        rng.shuffle(items)
        
        for b in items:
            redundant = True
            for tri_idx in block_triples[b]:
                if coverage_count[tri_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                for tri_idx in block_triples[b]:
                    coverage_count[tri_idx] -= 1
                selected.remove(b)
                removed = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                break
        
        if not removed:
            # 2. Try to swap: find a block with unique triples and replace it
            # Pick a random block from selected
            b = rng.choice(list(selected))
            
            # Find unique triples for b
            unique_tris = []
            for tri_idx in block_triples[b]:
                if coverage_count[tri_idx] == 1:
                    unique_tris.append(tri_idx)
            
            if not unique_tris:
                continue
            
            # Try to find a replacement that covers as many unique triples as possible
            # Pick the most constrained unique triple
            min_candidates = None
            best_tri = unique_tris[0]
            for tri in unique_tris:
                cands = blocks_containing[tri]
                if min_candidates is None or len(cands) < len(min_candidates):
                    min_candidates = cands
                    best_tri = tri
            
            cands = list(min_candidates)
            rng.shuffle(cands)
            
            swap_found = False
            for b_prime in cands:
                if b_prime in selected:
                    continue
                
                # Check how many unique triples b_prime covers
                b_prime_tris = block_triples_set[b_prime]
                covered = sum(1 for tri in unique_tris if tri in b_prime_tris)
                
                if covered == len(unique_tris):
                    # Full swap: covers all unique triples
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    
                    for tri_idx in block_triples[b_prime]:
                        coverage_count[tri_idx] += 1
                    selected.add(b_prime)
                    
                    swap_found = True
                    break
                elif covered >= len(unique_tris) - 1:
                    # Partial swap: covers all but one unique triple
                    # After swap, that one triple becomes uncovered
                    # We can try to fix it immediately
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    
                    for tri_idx in block_triples[b_prime]:
                        coverage_count[tri_idx] += 1
                    selected.add(b_prime)
                    
                    # Find the uncovered triple
                    uncovered_tri = None
                    for tri in unique_tris:
                        if tri not in b_prime_tris:
                            uncovered_tri = tri
                            break
                    
                    if uncovered_tri is not None and coverage_count[uncovered_tri] == 0:
                        # Try to find another block to cover it
                        fix_found = False
                        for b_fix in blocks_containing[uncovered_tri]:
                            if b_fix in selected:
                                continue
                            # Add b_fix to cover the gap
                            for tri_idx in block_triples[b_fix]:
                                coverage_count[tri_idx] += 1
                            selected.add(b_fix)
                            fix_found = True
                            break
                        
                        if not fix_found:
                            # Undo the swap
                            for tri_idx in block_triples[b_prime]:
                                coverage_count[tri_idx] -= 1
                            selected.remove(b_prime)
                            
                            for tri_idx in block_triples[b]:
                                coverage_count[tri_idx] += 1
                            selected.add(b)
                            continue
                    
                    swap_found = True
                    break
            
            if swap_found and len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)

    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # Safety margin for 120s limit
    
    # Precompute
    all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy + local search multiple times
    # We need to manage time carefully.
    # Let's do a fixed number of iterations, or until deadline.
    # Each iteration: Greedy (fast) + Local Search (variable).
    
    # To ensure we don't get stuck in one local search for 100s,
    # we can limit the local search time per iteration.
    # E.g., 5 seconds per iteration.
    
    while time.time() < deadline - 10: # Leave 10s for final check
        # 1. Greedy
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, block_triples, seed)
        seed += 1
        
        if not initial_blocks:
            continue
            
        # 2. Local Search
        # Give it a sub-deadline
        sub_deadline = min(deadline - 10, time.time() + 5)
        
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, sub_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # print(f"New best: {best_len} blocks at t={time.time()-start_time}")
            
    if best_blocks is None:
        # Fallback
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, 999)
        best_blocks = blocks
        
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
