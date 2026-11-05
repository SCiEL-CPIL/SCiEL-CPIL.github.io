import time
import random
from itertools import combinations

def construct():
    """Construct a packing of 6-subsets on 12 points with no repeated 5-subset.
    
    Returns a list of distinct 6-subsets of 0..11 such that no 5-subset lies in
    two blocks. Maximizes the number of blocks; the optimum 132 is attained exactly
    by the Steiner system S(5,6,12).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    # Precompute all 5-subsets and map to indices
    all_5subsets = list(combinations(range(n_points), subset_size))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    num_5subsets = len(all_5subsets)  # 792
    
    # Precompute all 6-subsets (blocks) and their 5-subset bitmasks
    all_6subsets = list(combinations(range(n_points), block_size))
    num_blocks = len(all_6subsets)  # 924
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for sub in combinations(b, subset_size):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Precompute for each block, the list of 5-subset indices it covers
    block_subsets = []
    for b in all_6subsets:
        subs = [subset_to_idx[sub] for sub in combinations(b, subset_size)]
        block_subsets.append(subs)
    
    target_mask = (1 << num_5subsets) - 1
    
    def greedy_packing(covered, order):
        """Greedy packing given an order of block indices."""
        blocks = []
        for idx in order:
            mask = block_masks[idx]
            if (covered & mask) == 0:
                blocks.append(idx)
                covered |= mask
                if len(blocks) == 132:
                    break
        return blocks
    
    # Strategy 1: Deterministic heuristics
    # Sort by sum of elements
    indices_by_sum = sorted(range(num_blocks), key=lambda i: sum(all_6subsets[i]))
    result = greedy_packing(0, indices_by_sum)
    if len(result) >= 132:
        return [list(all_6subsets[i]) for i in result]
    
    # Sort lexicographically
    indices_by_lex = sorted(range(num_blocks), key=lambda i: all_6subsets[i])
    result = greedy_packing(0, indices_by_lex)
    if len(result) >= 132:
        return [list(all_6subsets[i]) for i in result]
    
    # Strategy 2: Adaptive greedy - at each step, pick the block that covers
    # the most currently-uncovered 5-subsets (most-constrained-first)
    def adaptive_greedy():
        covered = 0
        blocks = []
        remaining = list(range(num_blocks))
        
        while remaining and len(blocks) < 132:
            # Find the uncovered 5-subset with fewest remaining blocks covering it
            # This is expensive, so we approximate: pick block with max new coverage
            best_idx = -1
            best_new = -1
            for idx in remaining:
                mask = block_masks[idx]
                new_bits = mask & ~covered
                # Count bits - use a fast approximation
                # Actually, let's just pick the one with most uncovered bits
                cnt = 0
                m = new_bits
                while m:
                    m &= m - 1
                    cnt += 1
                    if cnt > best_new:
                        break
                if cnt > best_new:
                    best_new = cnt
                    best_idx = idx
                    if cnt == 6:  # max possible
                        break
            
            if best_idx == -1 or best_new == 0:
                break
            
            blocks.append(best_idx)
            covered |= block_masks[best_idx]
            remaining.remove(best_idx)
        
        return blocks
    
    result = adaptive_greedy()
    if len(result) >= 132:
        return [list(all_6subsets[i]) for i in result]
    
    # Strategy 3: Random restart greedy with bitset optimization
    rng = random.Random(42)
    best_count = 0
    best_result = []
    
    for attempt in range(20000):
        if time.time() > deadline:
            break
        
        order = list(range(num_blocks))
        rng.shuffle(order)
        
        result = greedy_packing(0, order)
        
        if len(result) > best_count:
            best_count = len(result)
            best_result = result[:]
            if best_count >= 132:
                break
    
    if best_count >= 132:
        return [list(all_6subsets[i]) for i in best_result]
    
    # Fallback: return best found
    return [list(all_6subsets[i]) for i in best_result]
