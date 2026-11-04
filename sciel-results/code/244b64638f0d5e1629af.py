"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Precompute all t-subsets and assign them indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Greedy construction with better point selection
    # We'll use a bitmask approach for speed if feasible
    # For v=19, k=8, t=5: C(19,5) = 11628 t-subsets
    
    uncovered = set(range(n_tsubsets))  # indices of uncovered t-subsets
    blocks = []
    
    # Precompute for each point x, and each (t-1)-subset of remaining points,
    # which t-subsets would be covered
    # Actually, let's use a different approach:
    # For each potential block (k-subset), precompute which t-subsets it covers
    # But C(19,8) = 75582, which is manageable
    
    # Let's use a smarter greedy: at each step, try to find the best k-subset
    # that covers the most uncovered t-subsets
    
    # Precompute: for each k-subset, the list of t-subset indices it covers
    # This is C(19,8) * C(8,5) = 75582 * 56 ≈ 4.2M operations, feasible
    
    # Actually, let's be more careful. C(19,8) = 75582 blocks.
    # For each block, it covers C(8,5) = 56 t-subsets.
    # Total: 75582 * 56 = 4,232,592 entries. Manageable.
    
    # But storing all of them might be memory intensive. Let's compute on the fly
    # with caching.
    
    # Alternative: use a faster greedy with local search / hill climbing
    
    # Let's try a different approach: start with a good greedy, then do
    # local search to reduce the number of blocks.
    
    # Phase 1: Fast greedy construction
    def greedy_construct():
        uncovered_set = set(range(n_tsubsets))
        blocks = []
        
        while uncovered_set:
            # Pick the first uncovered t-subset (or a random one for diversity)
            # Use the lexicographically smallest for determinism in base case
            target_tsub = all_tsubsets[min(uncovered_set)]
            
            # Build a block around this t-subset
            # Start with the t-subset, add k-t more points greedily
            block = list(target_tsub)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                block_set = set(block)
                
                # For each candidate point not in block, compute gain
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain = number of uncovered t-subsets that contain x
                    # and are subsets of block ∪ {x}
                    # i.e., t-subsets S such that S ⊆ block ∪ {x}, 
                    # x ∈ S, and S is currently uncovered
                    
                    # Equivalent: count uncovered t-subsets S where 
                    # S \ {x} ⊆ block and x ∉ block
                    gain = 0
                    # t-subsets containing x with t-1 elements from block
                    # C(len(block), t-1) candidates
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_index.get(ts)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1 or best_gain == 0:
                    # No point adds coverage, just pick a random unused point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = random.choice(remaining)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                uncovered_set.discard(idx)
            
            if time.time() > deadline:
                break
        
        return blocks
    
    # Phase 1: Initial greedy
    blocks = greedy_construct()
    
    if time.time() > deadline:
        return blocks
    
    # Phase 2: Local search - try to remove redundant blocks
    # A block is redundant if every t-subset it covers is also covered by other blocks
    
    def is_redundant(block_idx, blocks):
        """Check if block at block_idx is redundant."""
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            idx = tsubset_index[ts]
            # Check if any other block covers this t-subset
            found = False
            for j, other_block in enumerate(blocks):
                if j == block_idx:
                    continue
                # Check if ts ⊆ other_block
                other_set = set(other_block)
                if all(x in other_set for x in ts):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(i, blocks):
                blocks.pop(i)
                changed = True
    
    if time.time() > deadline:
        return blocks
    
    # Phase 3: Try to replace blocks with better ones
    # For each block, try to find a replacement that covers more uncovered subsets
    # while maintaining coverage
    
    # Actually, let's try a different local search:
    # Randomly perturb and see if we can reduce the count
    
    best_blocks = blocks[:]
    best_count = len(blocks)
    
    # Hill climbing with restarts
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Start from current best
        current = best_blocks[:]
        
        # Try some random moves
        for move_iter in range(100):
            if time.time() > deadline:
                break
            
            if len(current) == 0:
                break
            
            # Move type 1: Remove a random block if coverage is maintained
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                removed_block = current[idx]
                current.pop(idx)
                
                # Check if coverage is still complete
                valid = True
                # We need to check if all t-subsets are covered
                # This is expensive, so let's be smarter
                # Only check t-subsets that were in the removed block
                for ts in itertools.combinations(removed_block, t):
                    ts_set = set(ts)
                    covered = False
                    for blk in current:
                        blk_set = set(blk)
                        if ts_set.issubset(blk_set):
                            covered = True
                            break
                    if not covered:
                        valid = False
                        break
                
                if valid:
                    best_blocks = current[:]
                    best_count = len(current)
                else:
                    current.insert(idx, removed_block)
            
            # Move type 2: Replace a block with a slightly different one
            if len(current) > 0 and random.random() < 0.3:
                idx = random.randint(0, len(current) - 1)
                old_block = current[idx][:]
                
                # Create a new block by swapping 1-2 points
                new_block = old_block[:]
                if len(new_block) > 0:
                    pos = random.randint(0, len(new_block) - 1)
                    old_point = new_block[pos]
                    candidates = [x for x in range(v) if x not in set(new_block)]
                    if candidates:
                        new_point = random.choice(candidates)
                        new_block[pos] = new_point
                        new_block.sort()
                        
                        # Check if coverage is maintained
                        valid = True
                        # Check t-subsets of old_block that might not be covered
                        for ts in itertools.combinations(old_block, t):
                            ts_set = set(ts)
                            covered = False
                            for j, blk in enumerate(current):
                                if j == idx:
                                    continue
                                blk_set = set(blk)
                                if ts_set.issubset(blk_set):
                                    covered = True
                                    break
                            if not covered:
                                valid = False
                                break
                        
                        if valid:
                            current[idx] = new_block
                            # Check if we can now remove another block
                            # (optional, for now just accept the swap)
                        
                        if time.time() > deadline:
                            break
        
        # After local search, check if we improved
        if len(current) < best_count:
            best_blocks = current[:]
            best_count = len(current)
    
    return best_blocks

# EVOLVE-BLOCK-END
