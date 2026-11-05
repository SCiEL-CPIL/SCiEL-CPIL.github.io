import time
import random
import sys
from itertools import combinations
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 118.0  # 118 seconds to be safe
    
    # 1. Pre-compute all t-subsets and map them to indices
    all_t_subsets = list(combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # 2. Pre-compute all possible blocks and their covered t-subsets as bitmasks
    all_blocks = list(combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store a bitmask of t-subsets it covers
    # Python ints are arbitrary precision, so we can use them as bitmasks
    # num_t = C(19,5) = 11628 bits, which fits in a Python int fine
    block_masks = []
    for b in all_blocks:
        mask = 0
        for ts in combinations(b, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # 3. Greedy Initialization with bitmask acceleration
    counts = np.zeros(num_t, dtype=np.int32)
    current_blocks = []
    block_indices = set()
    uncovered_mask = (1 << num_t) - 1  # All t-subsets uncovered initially
    
    while uncovered_mask and time_left() > 5.0:
        # Find block that covers most uncovered t-subsets
        # Use bitmask AND to compute intersection with uncovered
        best_block_idx = -1
        best_gain = -1
        for b_idx in range(num_blocks):
            gain = bin(block_masks[b_idx] & uncovered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_gain == 0:
            break
        
        current_blocks.append(best_block_idx)
        block_indices.add(best_block_idx)
        mask = block_masks[best_block_idx]
        # Update counts and uncovered_mask
        # For each bit set in mask, increment count
        # This is slow in Python for large masks, but num_blocks is ~75k and we do this ~300 times
        # Alternative: use numpy
        # Actually, let's use a different approach: keep counts as a list and update selectively
        # Since we only need to know if count is 0 or >0 for the uncovered mask,
        # and for redundancy checks, we can use the mask approach differently.
        
        # Let's use numpy for counts update
        # Get indices of set bits in mask
        # This is expensive. Let's precompute for each block the list of t-subset indices it covers.
        # We already have block_covers in the original. Let's keep that.
        
        # Actually, the bitmask approach for greedy is fast for the gain calculation,
        # but updating counts is still slow. Let's hybrid: use bitmask for gain,
        # but keep the list of covered indices for count updates.
        
        # Let me restructure: precompute both mask and list of indices
        pass
    
    # Let me restart with a cleaner hybrid approach
    # Precompute: for each block, the list of t-subset indices it covers
    block_covers = []
    for b in all_blocks:
        covers = []
        for ts in combinations(b, t):
            covers.append(t_to_idx[ts])
        block_covers.append(covers)
    
    # Now do greedy with bitmask for gain calculation
    counts = [0] * num_t
    current_blocks = []
    block_indices = set()
    uncovered = set(range(num_t))
    
    while uncovered and time_left() > 5.0:
        best_block_idx = -1
        best_gain = -1
        for b_idx in range(num_blocks):
            gain = bin(block_masks[b_idx] & ((1 << num_t) - 1)).count('1')  # This is wrong, need to mask with uncovered
            # Actually, I need to compute the intersection with uncovered properly
            # Let me use a different approach: just count how many of block_covers[b_idx] are in uncovered
            # But that's O(56) per block, which is 75k * 56 = 4.2M ops, which is fine.
            gain = sum(1 for ts_idx in block_covers[b_idx] if ts_idx in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_gain == 0:
            break
        
        current_blocks.append(best_block_idx)
        block_indices.add(best_block_idx)
        for ts_idx in block_covers[best_block_idx]:
            counts[ts_idx] += 1
            if counts[ts_idx] == 1:
                uncovered.discard(ts_idx)
    
    # 4. Prune Redundant Blocks
    def prune(blocks_list):
        new_blocks = []
        for b_idx in blocks_list:
            is_redundant = True
            for ts_idx in block_covers[b_idx]:
                if counts[ts_idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(b_idx)
            else:
                for ts_idx in block_covers[b_idx]:
                    counts[ts_idx] -= 1
                block_indices.discard(b_idx)
        return new_blocks
    
    current_blocks = prune(current_blocks)
    
    # 5. Local Search with Swap Operations
    best_blocks = list(current_blocks)
    best_len = len(best_blocks)
    
    # Precompute sets for faster membership checks
    block_covers_sets = [set(covers) for covers in block_covers]
    
    while time_left() > 2.0:
        # Try to remove a random redundant block
        if random.random() < 0.3 and current_blocks:
            b1_idx_pos = random.randint(0, len(current_blocks) - 1)
            b1_idx = current_blocks[b1_idx_pos]
            
            redundant = True
            for ts_idx in block_covers[b1_idx]:
                if counts[ts_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                for ts_idx in block_covers[b1_idx]:
                    counts[ts_idx] -= 1
                current_blocks.pop(b1_idx_pos)
                block_indices.discard(b1_idx)
                
                if len(current_blocks) < best_len:
                    best_blocks = list(current_blocks)
                    best_len = len(current_blocks)
        else:
            # Try to swap
            if not current_blocks:
                continue
            b1_idx_pos = random.randint(0, len(current_blocks) - 1)
            b1_idx = current_blocks[b1_idx_pos]
            
            b2_idx = random.randint(0, num_blocks - 1)
            if b2_idx in block_indices:
                continue
            
            covers_b1 = block_covers[b1_idx]
            covers_b2 = block_covers[b2_idx]
            covers_b2_set = block_covers_sets[b2_idx]
            
            valid = True
            for ts_idx in covers_b1:
                if counts[ts_idx] == 1:
                    if ts_idx not in covers_b2_set:
                        valid = False
                        break
            
            if valid:
                for ts_idx in covers_b1:
                    counts[ts_idx] -= 1
                for ts_idx in covers_b2:
                    counts[ts_idx] += 1
                
                current_blocks[b1_idx_pos] = b2_idx
                block_indices.discard(b1_idx)
                block_indices.add(b2_idx)
                
                # Prune
                current_blocks = prune(current_blocks)
                
                if len(current_blocks) < best_len:
                    best_blocks = list(current_blocks)
                    best_len = len(current_blocks)
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
