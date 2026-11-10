"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_masks = [0] * len(all_t_subsets)
    for i, sub in enumerate(all_t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_subset_masks[i] = mask
    
    # Map from mask to index for fast lookup? 
    # Actually, for greedy, we need to know which t-subsets are covered by a candidate block.
    # A block B covers a t-subset S if S is subset of B.
    # Number of t-subsets in a k-block is C(k, t).
    # For v=19, k=8, t=5: C(8,5)=56.
    # Total t-subsets: C(19,5)=11628.
    
    # We need to track which t-subsets are still uncovered.
    # Using a set of indices or a bitmask of t-subsets? 
    # 11628 bits is ~1.5KB. Python integers can handle this.
    # Let's represent the set of uncovered t-subsets as a Python integer bitmask.
    # Bit i is 1 if t-subset i is uncovered.
    
    def get_uncovered_mask():
        m = 0
        for i in range(len(all_t_subsets)):
            m |= (1 << i)
        return m

    # This is too slow to initialize every time if we do many restarts.
    # Let's precompute the full mask.
    full_uncovered_mask = 0
    for i in range(len(all_t_subsets)):
        full_uncovered_mask |= (1 << i)

    # Precompute for each possible k-block, the bitmask of t-subsets it covers?
    # There are C(19,8) = 75582 blocks.
    # For each block, we can precompute the mask of t-subsets it covers.
    # This takes C(19,8) * C(8,5) operations ~ 75k * 56 ~ 4M operations.
    # This is feasible and allows O(1) coverage check.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_mask = {}
    for b in all_blocks:
        mask = 0
        # Find all t-subsets of b
        for sub in itertools.combinations(b, t):
            # Need index of sub in all_t_subsets
            # This lookup is expensive if done naively.
            # Let's create a map from tuple to index.
            pass
            
    # Better: Create a map from t-subset tuple to its index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    for b in all_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_to_mask[b] = mask

    # Now greedy set cover:
    # uncovered_mask: integer
    # While uncovered_mask != 0:
    #   Find block B that maximizes popcount(uncovered_mask & block_to_mask[B])
    #   Add B to blocks
    #   uncovered_mask &= ~block_to_mask[B]
    
    # Optimizations:
    # 1. Only consider blocks that cover at least one uncovered subset.
    # 2. Random restarts: shuffle order or add noise to gain.
    # 3. Pruning: After greedy, try to remove blocks if their unique coverage is 0.
    
    best_blocks = None
    best_count = float('inf')
    
    # Since C(19,8) is small, we can afford to check all blocks in each step?
    # Steps ~ 300. 300 * 75k = 22.5M operations.
    # Bitwise AND and bit_count are fast in Python 3.8+.
    
    def greedy_cover(noise_seed=None):
        if noise_seed is not None:
            random.seed(noise_seed)
            
        uncovered = full_uncovered_mask
        blocks = []
        
        while uncovered:
            best_block = None
            best_gain = -1
            
            # Optimization: If time is tight, we might not be able to check all blocks
            # but 75k is small enough.
            
            for b in all_blocks:
                gain = (uncovered & block_to_mask[b]).bit_count()
                if gain > 0:
                    # Add small random noise to break ties and diversify
                    if noise_seed is not None:
                        gain += random.random() * 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
                        
            if best_block is None:
                # Should not happen if uncovered is not 0
                break
                
            blocks.append(best_block)
            uncovered &= ~block_to_mask[best_block]
            
            if time.time() > deadline:
                return None # Timeout
                
        return blocks

    def prune_blocks(blocks):
        # Remove redundant blocks
        # A block is redundant if all t-subsets it covers are covered by other blocks.
        # This is a hitting set / set cover reduction problem.
        # Simple greedy removal:
        # For each block, check if removing it leaves all its t-subsets covered by others.
        # This is O(N^2 * C(k,t)) which might be slow.
        # N ~ 300. N^2 = 90k. 90k * 56 = 5M ops. Feasible.
        
        # Precompute coverage of each block
        # covered_by[i] = set of block indices that cover t-subset i
        # Actually, we want to know for each block B, is there a t-subset S in B 
        # such that no other block B' in the list covers S?
        # If yes, B is essential.
        
        # Let's use a different approach:
        # 1. For each t-subset, find which blocks cover it.
        # 2. If a t-subset is covered by only one block, that block is essential.
        # 3. Remove non-essential blocks one by one, re-checking.
        
        # This is complex to implement efficiently within time.
        # Let's stick to a simpler heuristic: 
        # Sort blocks by "uniqueness" (number of t-subsets they uniquely cover).
        # Remove blocks with 0 unique t-subsets.
        
        # Calculate unique coverage
        # count_cover[i] = number of blocks covering t-subset i
        count_cover = [0] * len(all_t_subsets)
        block_indices = {b: i for i, b in enumerate(blocks)}
        
        for b in blocks:
            mask = block_to_mask[b]
            # Iterate through bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                count_cover[idx] += 1
                m ^= lsb
        
        # Identify essential blocks
        essential = [False] * len(blocks)
        for i, b in enumerate(blocks):
            mask = block_to_mask[b]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if count_cover[idx] == 1:
                    essential[i] = True
                    break
                m ^= lsb
        
        # Remove non-essential
        new_blocks = [blocks[i] for i in range(len(blocks)) if essential[i]]
        
        # Iterate pruning
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Recalculate count_cover for new_blocks
            count_cover = [0] * len(all_t_subsets)
            for b in new_blocks:
                mask = block_to_mask[b]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    count_cover[idx] += 1
                    m ^= lsb
            
            essential = [False] * len(new_blocks)
            for i, b in enumerate(new_blocks):
                mask = block_to_mask[b]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if count_cover[idx] == 1:
                        essential[i] = True
                        break
                    m ^= lsb
            
            old_len = len(new_blocks)
            new_blocks = [new_blocks[i] for i in range(len(new_blocks)) if essential[i]]
            if len(new_blocks) < old_len:
                changed = True
                
        return new_blocks

    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        blocks = greedy_cover(noise_seed=i)
        if blocks is None:
            continue
            
        # Prune
        blocks = prune_blocks(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we find a very good solution, we can stop early?
        # Best known is 335. If we get below 335, great.
        # Lower bound 264.
        
    if best_blocks is None:
        # Fallback to simple greedy if all failed (should not happen)
        blocks = greedy_cover(noise_seed=None)
        if blocks:
            return blocks
        return []
        
    return best_blocks

# EVOLVE-BLOCK-END
