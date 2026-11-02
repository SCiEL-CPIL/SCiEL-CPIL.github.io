"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Multi-start randomized greedy covering with redundancy removal.
    1. Precompute all t-subsets.
    2. Run multiple greedy constructions with different random seeds.
    3. For each construction, greedily pick blocks that cover the most uncovered t-subsets.
       If a block cannot be filled with positive gain, fill with random points to ensure validity.
    4. After construction, remove redundant blocks.
    5. Return the best (smallest) valid covering found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets for fast lookup
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    # We will run multiple restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        seed_val = restart_count * 12345 + 1
        random.seed(seed_val)
        
        blocks = _greedy_construct(v, k, t, all_t_subsets, deadline)
        
        if blocks is None:
            # Timeout during construction
            continue
            
        # Try to remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # If we reached a very good score, we can stop early to save time for safety
            # C(24,7,3) lower bound is 69, best known 76.
            if best_len <= 76:
                break

    if best_blocks is None:
        # Fallback: return an empty list or a trivial valid covering if we failed completely
        # But the contract says "Valid when every t-subset... lies in at least one block".
        # If we didn't finish, we must return a valid covering.
        # The greedy construction *always* produces a valid covering if it finishes.
        # If it timed out, we might not have a valid one.
        # To be safe, if we have no valid block set, we should do one final, fast, 
        # deterministic greedy pass that is guaranteed to finish quickly by being less aggressive.
        # However, the previous code failed because it didn't finish or produced invalid blocks.
        # My _greedy_construct ensures blocks are size k and covers are removed.
        # If it times out, it returns None.
        # We need a fallback that is *always* valid.
        # A trivial valid covering is all C(v,k) blocks, but that's huge.
        # Let's just run one more greedy pass with a very strict time check that ensures completion.
        # Actually, if best_blocks is None, it means we never completed a full greedy pass.
        # This is unlikely if the greedy pass is fast. 
        # To be absolutely safe, let's force a final construction with a smaller time slice 
        # or just return the last partial result if we can make it valid? No, partial is invalid.
        # Let's assume the first restart finishes. If not, we have a bug in time management.
        # Let's add a safety net: if best_blocks is None, run a simple, non-optimized greedy 
        # that doesn't check time inside the inner loop, but checks between blocks.
        # This might take longer but should finish.
        if time.time() < deadline - 5.0:
             blocks = _greedy_construct(v, k, t, all_t_subsets, deadline, force_fast=True)
             if blocks:
                 best_blocks = blocks
                 best_len = len(blocks)
        
        if best_blocks is None:
             # Last resort: return a list of blocks that covers everything? 
             # We can't generate C(24,7) quickly.
             # If we are here, the program is broken. 
             # But we must return *something*. 
             # Let's return an empty list to signal error? No, score 0.
             # Let's return a single block? Invalid.
             # We will assume best_blocks is not None.
             return []

    return best_blocks


def _greedy_construct(v, k, t, all_t_subsets, deadline, force_fast=False):
    """
    Greedy construction of a covering.
    Returns a list of blocks (tuples) or None if timeout.
    """
    uncovered = set(all_t_subsets)
    blocks = []
    
    # To speed up gain calculation, we can precompute which t-subsets contain which points?
    # Or just iterate.
    
    while uncovered:
        if time.time() > deadline:
            return None
            
        # Pick a random uncovered t-subset to be the seed of the new block
        # Using random helps in multi-start. 
        # If force_fast, use min for determinism and speed.
        if force_fast:
            seed_triple = min(uncovered)
        else:
            seed_triple = random.choice(list(uncovered))
            
        current_block = set(seed_triple)
        
        # Fill the block to size k
        while len(current_block) < k:
            if time.time() > deadline:
                return None
                
            best_point = None
            best_gain = -1
            best_gain_ties = []
            
            # Calculate gain for each candidate point
            # A point x adds gain equal to the number of uncovered t-subsets 
            # that contain x and (t-1) elements from current_block.
            
            # Optimization: Precompute (t-1) subsets of current_block
            # current_block size is at most k-1 = 6. t-1 = 2.
            # Number of (t-1) subsets is C(6, 2) = 15. Very small.
            t_minus_1_subsets = list(itertools.combinations(sorted(current_block), t - 1))
            
            for x in range(v):
                if x in current_block:
                    continue
                
                gain = 0
                for rest in t_minus_1_subsets:
                    # Construct the t-subset
                    # rest is a tuple of size t-1
                    # x is the new point
                    # We need to check if the sorted tuple is in uncovered
                    # To avoid sorting every time, we can construct a set or tuple
                    # Since t is small (3), sorting 3 elements is cheap.
                    t_subset = tuple(sorted(rest + (x,)))
                    if t_subset in uncovered:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_gain_ties = [x]
                elif gain == best_gain:
                    best_gain_ties.append(x)
                    
            if best_gain <= 0:
                # No point improves coverage of *uncovered* sets.
                # We must still fill the block to size k to make it a valid block.
                # Pick a random point not in the block.
                candidates = [x for x in range(v) if x not in current_block]
                if not candidates:
                    break # Should not happen if k <= v
                best_point = random.choice(candidates)
                # In force_fast, pick min
                if force_fast:
                    best_point = min(candidates)
            else:
                # Pick a random point from the best gain ties to add diversity
                best_point = random.choice(best_gain_ties)
                if force_fast:
                    best_point = min(best_gain_ties)
                    
            current_block.add(best_point)
            
        # Convert to sorted tuple for consistency
        block_tuple = tuple(sorted(current_block))
        blocks.append(block_tuple)
        
        # Update uncovered set
        # Remove all t-subsets of this block from uncovered
        block_covered = set(itertools.combinations(block_tuple, t))
        uncovered -= block_covered
        
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    n = len(blocks)
    if n > 300: # Skip if too large, redundancy removal is O(N * C(k,t))
        return blocks

    # Map each t-subset to the list of block indices that cover it
    subset_to_blocks = defaultdict(list)

    for i, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            subset_to_blocks[tuple(sub)].append(i)

    # A block is redundant if every t-subset it covers is also covered by another block
    keep = [True] * n
    for i, block in enumerate(blocks):
        if not keep[i]:
            continue
        is_redundant = True
        for sub in itertools.combinations(block, t):
            covering = subset_to_blocks[tuple(sub)]
            if len(covering) == 1 and covering[0] == i:
                # Only this block covers this subset
                is_redundant = False
                break
        if is_redundant:
            keep[i] = False

    new_blocks = [b for i, b in enumerate(blocks) if keep[i]]
    
    # If we removed all blocks (should not happen if input was valid covering),
    # return original to be safe
    if not new_blocks:
        return blocks
        
    return new_blocks
# EVOLVE-BLOCK-END
