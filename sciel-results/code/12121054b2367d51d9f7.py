import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # Safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # Precompute bitsets for each possible k-block (C(v,k) of them)
    # For v=24, k=7: C(24,7) = 346104 blocks, each with C(7,3)=35 t-subsets
    # We'll build a dictionary mapping block tuple -> bitset of covered t-subsets
    # But that's too many. Instead, we'll compute on-the-fly with caching.
    
    # Actually, let's use a different approach:
    # For each k-block, precompute which t-subsets it covers as a numpy boolean array
    # Or better: use Python integers as bitsets (Python has arbitrary precision ints)
    
    # Precompute the t-subsets that each potential block covers
    # We'll use a dict: block_tuple -> bitmask (int)
    # For C(24,7)=346104 blocks, each with 35 bits set, this is feasible
    
    block_bitmasks = {}
    for block in itertools.combinations(range(v), k):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            mask |= (1 << idx)
        block_bitmasks[block] = mask
    
    # Total number of t-subsets
    full_mask = (1 << n_tsubs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    max_restarts = 200  # Will be limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 7919 + 13)  # Different seed per restart
        
        covered = 0  # Bitmask of covered t-subsets
        blocks = []
        
        # Greedy phase: repeatedly pick the block that covers the most uncovered t-subsets
        while covered != full_mask:
            if time.time() > deadline - 3.0:
                break
            
            uncovered = full_mask ^ covered  # Bits that are still uncovered
            
            # Find the block that maximizes newly covered t-subsets
            # Sample a subset of candidate blocks to speed up
            # All possible blocks containing at least one uncovered t-subset
            
            # Strategy: pick a random uncovered t-subset, then consider all blocks containing it
            uncovered_indices = [i for i in range(n_tsubs) if not (covered >> i) & 1]
            if not uncovered_indices:
                break
            
            # Pick a random uncovered t-subset to focus on
            focus_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            focus_tsub = all_tsubs[focus_idx]
            
            # Find all k-blocks containing focus_tsub
            remaining = [x for x in range(v) if x not in focus_tsub]
            candidates = []
            for combo in itertools.combinations(remaining, k - t):
                block = tuple(sorted(focus_tsub + combo))
                candidates.append(block)
            
            # Among candidates, find the one with maximum new coverage
            best_block = None
            best_gain = -1
            
            for block in candidates:
                mask = block_bitmasks[block]
                gain = bin(mask & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            covered |= block_bitmasks[best_block]
        
        # Check if we have a complete covering
        if covered == full_mask:
            # Local search: try to remove redundant blocks
            # A block is redundant if all its t-subsets are covered by other blocks
            final_blocks = list(blocks)
            
            # Fast redundancy removal using reference counts
            # Count how many times each t-subset is covered
            tsub_counts = np.zeros(n_tsubs, dtype=np.int32)
            block_tuples = [tuple(b) for b in final_blocks]
            for bt in block_tuples:
                mask = block_bitmasks[bt]
                # Extract set bits from mask
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    tsub_counts[idx] += 1
                    m ^= lsb
            
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                # Find blocks where every t-subset they cover is also covered by others
                for i in range(len(final_blocks) - 1, -1, -1):
                    if time.time() > deadline - 1.0:
                        break
                    
                    bt = block_tuples[i]
                    mask = block_bitmasks[bt]
                    # Check if all t-subsets in this block have count > 1
                    m = mask
                    redundant = True
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if tsub_counts[idx] <= 1:
                            redundant = False
                            break
                        m ^= lsb
                    
                    if redundant:
                        # Remove this block and update counts
                        m = mask
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            tsub_counts[idx] -= 1
                            m ^= lsb
                        final_blocks.pop(i)
                        block_tuples.pop(i)
                        improved = True
                        break  # Restart the loop as coverage changed
            
            # Block-swap local search: try replacing one block with another to maintain coverage
            # This can reduce total count if we find a block that covers more unique t-subsets
            if time.time() < deadline - 2.0:
                swap_improved = True
                while swap_improved and time.time() < deadline - 2.0:
                    swap_improved = False
                    for i in range(len(final_blocks)):
                        if time.time() > deadline - 2.0:
                            break
                        bt_i = block_tuples[i]
                        mask_i = block_bitmasks[bt_i]
                        # t-subsets only covered by block i
                        unique_to_i = mask_i
                        for j in range(len(final_blocks)):
                            if j != i:
                                unique_to_i &= ~block_bitmasks[block_tuples[j]]
                        
                        if unique_to_i == 0:
                            continue
                        
                        # Try to find a replacement block that covers unique_to_i
                        # and has good overall coverage
                        # Sample some candidate blocks
                        for _ in range(50):  # Try a few random candidates
                            if time.time() > deadline - 2.0:
                                break
                            # Pick a random point set that covers unique_to_i
                            # We need a k-block that contains all t-subsets in unique_to_i
                            # This is complex, so instead try swapping with a random block
                            # that covers the unique t-subsets
                            
                            # Get one t-subset from unique_to_i
                            m = unique_to_i
                            if m == 0:
                                break
                            lsb = m & (-m)
                            focus_idx = lsb.bit_length() - 1
                            focus_tsub = all_tsubs[focus_idx]
                            
                            remaining = [x for x in range(v) if x not in focus_tsub]
                            # Try a random completion
                            if len(remaining) >= k - t:
                                combo = rng.sample(remaining, k - t)
                                new_block = tuple(sorted(list(focus_tsub) + combo))
                                
                                if new_block in [bt for bt in block_tuples]:
                                    continue
                                
                                new_mask = block_bitmasks[new_block]
                                # Check if new_block covers all of unique_to_i
                                if (new_mask & unique_to_i) != unique_to_i:
                                    continue
                                
                                # Check coverage after swap
                                # Remove block i, add new_block
                                new_covered = 0
                                for j, bt in enumerate(block_tuples):
                                    if j == i:
                                        new_covered |= new_mask
                                    else:
                                        new_covered |= block_bitmasks[bt]
                                
                                if new_covered == full_mask:
                                    # Do the swap
                                    final_blocks[i] = list(new_block)
                                    block_tuples[i] = new_block
                                    swap_improved = True
                                    break
                        
                        if swap_improved:
                            break
            
            if len(final_blocks) < best_count:
                best_count = len(final_blocks)
                best_blocks = final_blocks
        
        # Even if not complete, keep track for potential use
        # But we need a valid covering, so only complete ones count
    
    # If no complete covering found (shouldn't happen), return empty or best effort
    if best_blocks is None:
        # Fallback: greedy without local search, should always complete
        covered = 0
        blocks = []
        rng = random.Random(42)
        while covered != full_mask:
            uncovered_indices = [i for i in range(n_tsubs) if not (covered >> i) & 1]
            if not uncovered_indices:
                break
            focus_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            focus_tsub = all_tsubs[focus_idx]
            remaining = [x for x in range(v) if x not in focus_tsub]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(remaining, k - t):
                block = tuple(sorted(focus_tsub + combo))
                mask = block_bitmasks[block]
                gain = bin(mask & (full_mask ^ covered)).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            covered |= block_bitmasks[best_block]
        best_blocks = blocks
        best_count = len(blocks)
    
    return best_blocks
