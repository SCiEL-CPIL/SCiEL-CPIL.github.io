import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets and map to IDs
    # For v=19, k=8, t=5, C(19,5) = 11628
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    # covered[i] is True if subset i is covered by at least one block
    covered = np.zeros(n_subsets, dtype=bool)
    # cover_count[i] is the number of blocks covering subset i
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    
    blocks = []
    
    # Helper to update cover counts
    def update_cover(block, delta):
        for sub in itertools.combinations(block, t):
            idx = subset_to_id[sub]
            cover_count[idx] += delta
            if delta > 0 and cover_count[idx] == 1:
                covered[idx] = True
            elif delta < 0 and cover_count[idx] == 0:
                covered[idx] = False
                
    # Phase 1: Greedy Construction
    # We need to pick points to form blocks that cover many uncovered subsets.
    # Gain of adding point x to current block B:
    # Number of uncovered subsets T such that T \ {x} is a subset of B.
    # Since |T|=t, T \ {x} has size t-1.
    # So we look for (t-1)-subsets Q of B such that Q U {x} is uncovered.
    
    while not covered.all():
        if time.time() > deadline:
            # Return best effort (which is valid if we only added blocks, 
            # but greedy ensures we stop when covered.all() is True, 
            # so if we break early, it might be invalid. 
            # However, greedy adds blocks one by one. 
            # If we run out of time, we must return a VALID cover.
            # The loop condition `while not covered.all()` ensures we only exit when valid.
            # If we break early, we should probably just return what we have 
            # IF it's valid, otherwise we have a problem.
            # But the greedy step is fast. Let's assume it finishes.
            # To be safe, if we break, we must ensure validity. 
            # Actually, if we break, `covered` might not be all True. 
            # We cannot return an invalid cover. 
            # So we must complete the greedy phase. 
            # Greedy phase is O(N_blocks * k * C(k, t-1)). 
            # N_blocks ~ 300. k=8. C(7,4)=35. 300*8*35 = 84000 ops. Very fast.
            pass

        # Start a new block
        # Pick a random point to start, or the point in the most uncovered subset?
        # Let's pick the first point of the lexicographically first uncovered subset.
        # Or just 0.
        # Better: Pick a point that appears in many uncovered subsets?
        # For simplicity, start with 0.
        block = [0]
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            # Calculate gain for each x not in block
            # Gain(x) = number of Q in C(block, t-1) such that Q U {x} is uncovered
            # Note: If t-1 > len(block), gain is 0.
            if len(block) >= t - 1:
                # Precompute the (t-1) subsets of the current block
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                
                # For efficiency, we can iterate candidates and compute gain
                # But C(block, t-1) is small.
                t_minus_1_subs = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for Q in t_minus_1_subs:
                        T = tuple(sorted(Q + (x,)))
                        if T in subset_to_id:
                            idx = subset_to_id[T]
                            if not covered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No gain? Just pick a random point to fill the block
                # This can happen if the block is "isolated" from uncovered sets
                # but we need to form a valid block of size k.
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break # Should not happen if k <= v
            
            block.append(best_x)
            
        # Add the block
        block.sort()
        blocks.append(block)
        update_cover(block, 1)
        
    # Phase 2: Local Search (Swap and Shrink)
    # Try to reduce the number of blocks.
    
    def is_block_redundant(block_idx):
        # Check if all t-subsets of block_idx are covered by other blocks
        block = blocks[block_idx]
        for sub in itertools.combinations(block, t):
            idx = subset_to_id[sub]
            if cover_count[idx] == 1:
                return False
        return True

    def try_swap(block_idx, p, q):
        # Check if swapping p for q in block_idx keeps the cover valid
        # Only need to check subsets that lose coverage:
        # T in C(block, t) such that p in T and q not in T.
        block = blocks[block_idx]
        # Construct the list of subsets to check
        # These are subsets of (block - {p}) of size t-1, union {p}
        # Wait, T must contain p. So T = {p} U Q, where Q is a subset of (block - {p}) of size t-1.
        # And we require q NOT in T. Since q is not in block, q is not in Q.
        # So we just check all T = {p} U Q for Q in C(block - {p}, t-1).
        
        block_without_p = [x for x in block if x != p]
        
        for Q in itertools.combinations(block_without_p, t - 1):
            T = tuple(sorted(Q + (p,)))
            idx = subset_to_id[T]
            # This subset T is currently covered by block_idx.
            # If we swap, block_idx no longer covers T.
            # So we need cover_count[idx] > 1 (so it's covered by someone else)
            if cover_count[idx] == 1:
                return False
        return True

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Step 1: Remove redundant blocks
        # Iterate backwards to avoid index shifting issues
        for i in range(len(blocks) - 1, -1, -1):
            if is_block_redundant(i):
                # Remove block i
                block = blocks[i]
                update_cover(block, -1)
                blocks.pop(i)
                improved = True
        
        # Step 2: Try swaps
        if not improved:
            # Try to find a beneficial swap
            # We don't have a clear "benefit" metric other than enabling future removals.
            # So we just try to find ANY valid swap, accept it, and restart the loop.
            # This is a random walk in the space of valid covers.
            # To make it deterministic or efficient, we can try all swaps and pick one.
            
            # Optimization: Only try swaps for blocks that are "critical" or random ones?
            # Let's try all swaps in a random order.
            
            # To avoid getting stuck in a loop of swaps that don't lead to removals,
            # we can limit the number of swaps per pass or use a counter.
            
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                block = blocks[i]
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                
                # Shuffle candidates to randomize search
                random.shuffle(candidates)
                
                for p in block:
                    for q in candidates:
                        if try_swap(i, p, q):
                            # Perform swap
                            # Update cover counts for the block
                            old_block = block.copy()
                            new_block = [x if x != p else q for x in block]
                            new_block.sort()
                            
                            # Subtract old block contributions
                            update_cover(old_block, -1)
                            # Add new block contributions
                            update_cover(new_block, 1)
                            
                            # Update blocks list
                            blocks[i] = new_block
                            
                            improved = True
                            break # Restart the whole process (shrink phase)
                    if improved:
                        break
                if improved:
                    break

    return blocks

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the required structure if necessary, 
# but the task says "Keep the function names and signatures".
# The seed had `def construct(v, k, t):`.
# I will output the complete program.
# EVOLVE-BLOCK-END
