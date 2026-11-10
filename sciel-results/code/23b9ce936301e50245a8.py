import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each t-subset, the list of indices of points NOT in the subset
    # This allows faster generation of candidate blocks
    # For each t-subset s (tuple of t points), remaining_points are points not in s
    # We precompute these lists for speed
    
    # Precompute all possible blocks and the t-subsets they cover
    # A block is a k-subset of v points
    # For v=19, k=8, C(19,8) = 75582 blocks
    # Each block covers C(8,5) = 56 t-subsets
    
    # Instead of precomputing all blocks, let's use a smarter approach:
    # For each t-subset, precompute the list of all blocks containing it
    # But that's memory intensive. Let's use a different strategy.
    
    # Strategy: Use a bitset representation for uncovered t-subsets
    # We can use Python integers as bitsets (arbitrary precision)
    # Each t-subset has an index 0..num_t_subsets-1
    # A block covers a set of t-subsets, which we can represent as a bitmask
    
    # Precompute for each possible block (k-subset), the bitmask of t-subsets it covers
    # But there are C(19,8) = 75582 blocks, and each bitmask is 116280 bits = 14.5 KB
    # Total memory: 75582 * 14.5 KB ~ 1.1 GB. Too much.
    
    # Alternative: Don't precompute all blocks. Instead, when we pick a t-subset s,
    # generate all blocks containing s on the fly, and compute their coverage mask.
    
    # Let's precompute for each t-subset s, the list of "extension points" (points not in s)
    # and cache the coverage masks for blocks we've seen
    
    best_blocks = None
    best_len = float('inf')
    
    points = list(range(v))
    
    def greedy_once(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        # Bitset of uncovered t-subsets
        # Initially all 1s
        uncovered_mask = (1 << num_t_subsets) - 1
        blocks = []
        
        # Cache for block -> coverage mask
        # Key: block (tuple), Value: mask (int)
        block_mask_cache = {}
        
        while uncovered_mask != 0:
            # Pick a t-subset from uncovered.
            # Heuristic: Pick one that appears in fewest potential blocks? 
            # Or just pick a random uncovered t-subset.
            
            # To pick a random uncovered t-subset, we can find a set bit
            # Let's pick a random position and check if it's uncovered, or iterate
            # For efficiency, let's maintain a list of uncovered t-subsets
            # But that's expensive to update. Let's just pick a random index and check.
            
            # Actually, let's maintain a list of uncovered t-subset indices
            # and pick randomly from it
            uncovered_indices = []
            # This is O(num_t_subsets) each time, which is 116280 * 335 steps = 39M ops. Too slow.
            
            # Better: Just pick a random t-subset and check if it's uncovered.
            # If not, retry. But this could be slow if few are uncovered.
            
            # Alternative: Use the fact that we're doing greedy, so the "most constrained" 
            # t-subset is a good heuristic. But computing that is expensive.
            
            # Let's just pick a random t-subset that is uncovered.
            # We can do this by generating random indices until we find one that's uncovered.
            
            s_idx = None
            max_attempts = 1000
            for _ in range(max_attempts):
                cand_idx = rng.randrange(num_t_subsets)
                if (uncovered_mask >> cand_idx) & 1:
                    s_idx = cand_idx
                    break
            
            if s_idx is None:
                # Fallback: find the first uncovered t-subset
                for i in range(num_t_subsets):
                    if (uncovered_mask >> i) & 1:
                        s_idx = i
                        break
                if s_idx is None:
                    break
            
            s = all_t_subsets[s_idx]
            
            # Find all blocks of size k that contain s
            s_set = set(s)
            remaining_points = [p for p in points if p not in s_set]
            
            best_block = None
            best_gain = -1
            
            # Generate all combinations of k-t points from remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(s + combo))
                
                # Get or compute the coverage mask for this block
                if block in block_mask_cache:
                    mask = block_mask_cache[block]
                else:
                    # Compute mask: for each t-subset of block, set the corresponding bit
                    mask = 0
                    for sub in itertools.combinations(block, t):
                        idx = t_subset_to_idx[sub]
                        mask |= (1 << idx)
                    block_mask_cache[block] = mask
                
                # Gain = number of uncovered t-subsets covered by this block
                # = popcount(uncovered_mask & mask)
                gain = bin(uncovered_mask & mask).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_block_mask = mask
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_block = block
                        best_block_mask = mask
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Remove covered t-subsets from uncovered
            uncovered_mask &= ~best_block_mask
            
            # Time check
            if time.time() - start_time > time_limit - 5.0:
                return None
        
        return blocks

    def local_search(blocks, random_seed=None):
        """Try to remove redundant blocks from a valid covering."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        # Convert blocks to a set for quick lookup
        block_set = set(blocks)
        n = len(blocks)
        
        # For each t-subset, count how many blocks cover it
        # and track which blocks cover it
        t_subset_cover_count = [0] * num_t_subsets
        t_subset_coverers = [set() for _ in range(num_t_subsets)]
        
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                t_subset_cover_count[idx] += 1
                t_subset_coverers[idx].add(id(block))  # Use id for speed, but need to be careful
        
        # Actually, let's use the block itself as key, but that's expensive.
        # Let's use a different approach: for each block, check if it's the only 
        # block covering some t-subset. If not, try to remove it.
        
        # Better approach: For each block, compute the "exclusive" t-subsets it covers
        # (t-subsets that no other block covers). If a block has no exclusive t-subsets,
        # it can be removed.
        
        # But computing this is expensive. Let's use a simpler heuristic:
        # Try to remove each block in random order, and check if the covering is still valid.
        
        # For speed, we'll use a bitset approach.
        # Precompute the mask for each block
        block_masks = []
        for block in blocks:
            mask = 0
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                mask |= (1 << idx)
            block_masks.append(mask)
        
        # The union of all masks should cover all t-subsets
        # Try to remove blocks one by one
        new_blocks = list(blocks)
        new_masks = list(block_masks)
        
        # Compute the current union
        union_mask = 0
        for m in new_masks:
            union_mask |= m
        
        # For each block, check if removing it still leaves full coverage
        # A block can be removed if union_mask without it still covers all t-subsets
        # i.e., (union_mask & ~block_mask) still has all bits set
        # But we need to check if the union of the OTHER blocks covers all t-subsets.
        
        # This is expensive to check for each block. Let's use a different approach:
        # Count how many blocks cover each t-subset. If any t-subset is covered by 
        # exactly one block, that block cannot be removed.
        
        # Let's count coverage
        cover_count = [0] * num_t_subsets
        for m in new_masks:
            # Count bits in m
            # We need to add 1 to cover_count for each set bit in m
            # This is expensive. Let's use a different strategy.
            
            # For now, let's just try random removals and validate
            pass
        
        # Simpler local search: try to swap a block with a better one
        # Or just try to remove blocks that are "redundant"
        
        # For time efficiency, let's do a simple pass:
        # For each block (in random order), check if it's the only block covering 
        # some t-subset. If not, remove it.
        
        # To do this efficiently, we need to know for each t-subset, how many blocks cover it.
        # Let's build this count.
        
        # Since we have the masks, we can count as follows:
        # For each t-subset index i, count how many masks have bit i set.
        # This is O(n * num_t_subsets / word_size) if we use bit operations.
        
        # For now, let's skip local search for simplicity and rely on more restarts.
        # The key improvement is the bitset-based greedy which should be faster.
        
        return blocks

    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < time_limit - 2.0:
        seed = num_restarts
        res = greedy_once(seed)
        num_restarts += 1
        
        if res is not None:
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
        
        if best_len <= 264:
            break

    if best_blocks is None:
        if time.time() - start_time < time_limit - 5.0:
             best_blocks = greedy_once(0)
    
    return best_blocks
