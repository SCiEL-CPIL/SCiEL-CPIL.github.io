import itertools
import random
import time

# Pre-computation cache
_cache = {}

def _precompute(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]
    
    # 1. Generate all t-subsets and assign them indices
    t_subs = list(itertools.combinations(range(v), t))
    num_t = len(t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # 3. For each k-block, compute the bitmask of t-subsets it covers
    block_masks = []
    for b in k_blocks:
        mask = 0
        # A k-block covers all t-subsets within it
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # 4. For each t-subset, list the indices of k-blocks that cover it
    # This is used for the "pivot" strategy
    subset_to_blocks = [[] for _ in range(num_t)]
    for b_idx, mask in enumerate(block_masks):
        # Iterate through set bits in mask to find which t-subsets are covered
        # This is O(num_t) per block, which is 11628 * 75582 ~ 875M ops. Too slow.
        pass
        
    # Better way to build subset_to_blocks:
    # Iterate through k_blocks, and for each, iterate through its t-subsets
    for b_idx, b in enumerate(k_blocks):
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            subset_to_blocks[idx].append(b_idx)
            
    _cache[(v, k, t)] = (k_blocks, block_masks, subset_to_blocks, num_t)
    return _cache[(v, k, t)]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    k_blocks, block_masks, subset_to_blocks, num_t = _precompute(v, k, t)
    
    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy iterations with different random pivots
    num_iterations = 50
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        rng = random.Random(i)
        uncovered = (1 << num_t) - 1
        blocks_used = []
        
        while uncovered:
            # Find a random uncovered t-subset to use as pivot
            # Get index of a random set bit
            # Method: get a random number 0..num_t-1, check if set. If not, try again?
            # Or: convert to list of set bits? Too slow.
            # Better: Pick a random integer r, compute (uncovered >> r) & 1.
            # But we need a guaranteed set bit.
            # Efficient way:
            # 1. Count bits? No.
            # 2. Just pick a random index and check. If not set, pick another.
            # Since density is high initially, this is fast. Later, density is low, but num_t is small.
            # Actually, we can just pick the lowest set bit for determinism in the first few, 
            # or randomize by shuffling the order of checks?
            
            # Let's use a simpler randomization:
            # Pick a random offset, rotate the bitset? No.
            # Just pick a random index 0..num_t-1. If it's covered, try next.
            # To avoid infinite loop if uncovered is sparse, we can maintain a list of uncovered indices?
            # That list updates are expensive.
            
            # Alternative: Use the lowest set bit, but add a random tie-breaker in block selection?
            # Or, simply: Pick a random index. If not in uncovered, decrement index until found?
            # This is O(num_t) worst case.
            
            # Let's stick to: Pick a random index. If not set, try a few times.
            # If too many failures, fall back to lowest set bit.
            pivot_idx = -1
            for _ in range(10):
                cand = rng.randint(0, num_t - 1)
                if (uncovered >> cand) & 1:
                    pivot_idx = cand
                    break
            
            if pivot_idx == -1:
                # Fallback to lowest set bit
                lsb = uncovered & -uncovered
                pivot_idx = lsb.bit_length() - 1
                
            # Get candidate blocks for this pivot
            candidates = subset_to_blocks[pivot_idx]
            
            best_block_idx = -1
            max_gain = 0
            
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = b_idx
                    
            # It's possible max_gain is 0? No, because the pivot is uncovered, 
            # and all candidates cover the pivot, so gain >= 1.
            
            blocks_used.append(k_blocks[best_block_idx])
            uncovered &= ~block_masks[best_block_idx]
            
        if len(blocks_used) < best_len:
            best_len = len(blocks_used)
            best_solution = blocks_used
            
    return best_solution
