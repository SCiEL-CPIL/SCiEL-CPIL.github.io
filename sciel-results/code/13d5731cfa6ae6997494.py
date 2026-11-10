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
        # Remove redundant blocks using iterative essential block removal
        # and local search (swap-based optimization)
        
        if len(blocks) <= 1:
            return list(blocks)
        
        def is_valid_cover(blk_list):
            """Check if blk_list covers all t-subsets."""
            covered = 0
            for b in blk_list:
                covered |= block_to_mask[b]
            return covered == full_uncovered_mask
        
        def try_remove_redundant(blk_list):
            """Remove blocks that are not essential (no unique t-subset coverage)."""
            if not blk_list:
                return blk_list
            
            count_cover = [0] * len(all_t_subsets)
            for b in blk_list:
                mask = block_to_mask[b]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    count_cover[idx] += 1
                    m ^= lsb
            
            essential = [False] * len(blk_list)
            for i, b in enumerate(blk_list):
                mask = block_to_mask[b]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if count_cover[idx] == 1:
                        essential[i] = True
                        break
                    m ^= lsb
            
            new_blocks = [blk_list[i] for i in range(len(blk_list)) if essential[i]]
            return new_blocks
        
        def local_search(blk_list):
            """Try to replace blocks with better ones to reduce count."""
            if not blk_list:
                return blk_list
            
            current = list(blk_list)
            improved = True
            max_iter = 50  # Limit iterations
            
            while improved and time.time() < deadline and max_iter > 0:
                improved = False
                max_iter -= 1
                
                n = len(current)
                if n < 2:
                    break
                
                # For each pair of blocks, try swapping one out for a different block
                # that covers more uncovered subsets when the first is removed
                for i in range(n):
                    if time.time() > deadline:
                        return current
                    
                    # Temporarily remove block i
                    temp_list = current[:i] + current[i+1:]
                    temp_covered = 0
                    for b in temp_list:
                        temp_covered |= block_to_mask[b]
                    
                    # Find uncovered subsets after removing block i
                    uncovered_after = full_uncovered_mask & ~temp_covered
                    if uncovered_after == 0:
                        # Block i is redundant, remove it
                        current = temp_list
                        improved = True
                        break
                    
                    # Try to find a replacement block that covers more of uncovered_after
                    best_repl = None
                    best_gain = 0
                    for b in all_blocks:
                        if b in current:
                            continue
                        gain = (uncovered_after & block_to_mask[b]).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_repl = b
                    
                    if best_repl is not None and best_gain > 0:
                        # Check if replacement actually helps (covers at least as much as removed)
                        removed_gain = (uncovered_after & block_to_mask[current[i]]).bit_count()
                        if best_gain >= removed_gain:
                            current[i] = best_repl
                            improved = True
                            break
            
            return current
        
        # Phase 1: Remove non-essential blocks iteratively
        current = list(blocks)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            old_len = len(current)
            current = try_remove_redundant(current)
            if len(current) < old_len:
                changed = True
        
        # Phase 2: Local search to try to reduce further
        if time.time() < deadline:
            current = local_search(current)
        
        # Phase 3: Final cleanup of any newly redundant blocks
        if time.time() < deadline:
            current = try_remove_redundant(current)
        
        return current

    # Run multiple restarts with adaptive time allocation
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        blocks = greedy_cover(noise_seed=i)
        if blocks is None:
            continue
            
        # Prune with local search
        blocks = prune_blocks(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Early stop if we reach the known best or better
        if best_count <= 335:
            break
            
    if best_blocks is None:
        # Fallback to simple greedy if all failed (should not happen)
        blocks = greedy_cover(noise_seed=None)
        if blocks:
            return blocks
        return []
        
    return best_blocks

# EVOLVE-BLOCK-END
