import itertools
import numpy as np
import time
import sys

# Precomputation for v=19, k=8, t=5
# We will compute these inside the function or globally if we assume fixed params.
# The task says "evaluator passes this task's parameters v, k and t".
# So we must handle dynamic v, k, t.
# However, C(19,8,5) is the specific hard case.
# We will write a general solution but optimized for the expected small v.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Precompute all t-subsets and their indices
    # For v=19, t=5, there are 11628 subsets.
    # We map each subset (tuple of sorted points) to an integer index.
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_sets = len(t_subsets)
    
    # Mapping from tuple to index
    # Using a dictionary for O(1) lookup
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # For fast gain calculation, we might want to know, for a point x and a set R (subset of block),
    # the index of R U {x}.
    # But R varies.
    
    # Uncovered mask
    uncovered = np.ones(n_t_sets, dtype=bool)
    
    blocks = []
    
    # Helper to update uncovered mask for a block
    def update_uncovered(block):
        # Mark all t-subsets of block as covered
        # block is a list of points
        # We need to find all t-subsets of block
        # And mark their indices in 'uncovered' as False
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            uncovered[idx] = False

    # Helper to calculate gain of adding x to block B
    def calc_gain(block, x):
        # Gain is number of uncovered t-sets T such that T subset of B U {x} and x in T
        # This implies T \ {x} is a subset of B of size t-1
        # So we iterate over all (t-1)-subsets R of B
        # Check if R U {x} is in subset_to_idx (it always is if x not in R and size correct)
        # And if it is uncovered
        gain = 0
        # If len(block) < t-1, gain is 0
        if len(block) < t - 1:
            return 0
        for R in itertools.combinations(block, t - 1):
            # R is a tuple of t-1 points
            # We need to form T = R U {x}
            # Sort it to match the key in subset_to_idx
            # Since R is from combinations, it's sorted.
            # We just need to insert x and sort.
            # For speed, we can construct the tuple manually if we know R is sorted.
            # But simple tuple(sorted(...)) is fine for small t.
            T = tuple(sorted(R + (x,)))
            # Check if T is a valid t-subset (it is, by construction, if x not in R)
            # And if it is uncovered
            if T in subset_to_idx:
                idx = subset_to_idx[T]
                if uncovered[idx]:
                    gain += 1
        return gain

    # Greedy Phase
    while True:
        if time.time() > deadline - 5.0:
            break
            
        # Find first uncovered set
        # We need to find an index i where uncovered[i] is True
        # np.nonzero(uncovered) returns array of indices
        unc_idx = np.nonzero(uncovered)[0]
        if len(unc_idx) == 0:
            break
        
        # Start block with first point of the first uncovered set
        first_unc_set = t_subsets[unc_idx[0]]
        block = [first_unc_set[0]]
        
        # Build block
        while len(block) < k:
            if time.time() > deadline - 5.0:
                # If we run out of time, we might have an incomplete block.
                # We should probably discard the current block if it's not valid?
                # But we haven't added it to 'blocks' yet.
                # So we just break out of the while loop and return what we have.
                # But 'blocks' only contains completed blocks.
                # So we just return.
                break
                
            best_x = -1
            best_gain = -1
            
            # Candidates: points not in block
            # To speed up, we can just iterate 0..v-1
            for x in range(v):
                if x in block:
                    continue
                g = calc_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_gain == 0:
                # No point adds coverage? This shouldn't happen if uncovered is not empty
                # and we started with a point from an uncovered set.
                # But if the uncovered set requires points not in block yet?
                # Wait, we start with one point.
                # If t=5, we need 4 more points from the set.
                # calc_gain checks subsets of size t-1=4.
                # If block has size 1, we need 4 points from block to form R?
                # No, R is subset of block.
                # If block size is 1, we can't form R of size 4.
                # So gain is 0 for all x.
                # This logic is flawed for the first few steps.
                
                # Correction:
                # The gain of x given block B is the number of t-sets T
                # such that T \ {x} \subseteq B.
                # This requires |B| >= t-1.
                # If |B| < t-1, the gain is 0.
                # So we should just pick an arbitrary point if gain is 0?
                # No, we should pick a point that is in an uncovered set containing B.
                # Better heuristic:
                # If |B| < t-1, we want to pick x such that it is in as many uncovered sets
                # that contain B as possible.
                # i.e., count T in uncovered such that B \subseteq T and x \in T.
                
                # Let's refine calc_gain to handle |B| < t-1.
                # If |B| < t-1:
                # Gain(x) = count of T in uncovered such that B \subseteq T and x \in T.
                # This is equivalent to:
                # Iterate over all T in uncovered? Too slow.
                # Precompute for each point x, the list of T containing x?
                # And check if B \subseteq T.
                
                # Let's rewrite calc_gain to be robust.
                pass
            
            if best_x == -1:
                # Fallback: pick any x not in block
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
            
            block.append(best_x)
        
        # If we broke out due to time, block might be incomplete or just finished.
        # We only add to blocks if it has size k?
        # The task requires blocks of size k.
        # If we didn't finish, we should probably not add it, or add a random completion?
        # Adding a random completion might be bad.
        # But if we don't add it, we might leave sets uncovered.
        # However, if we run out of time, we return what we have.
        # The evaluator checks if EVERY t-subset is covered.
        # If we don't add the block, it's not covered.
        # So we MUST ensure we finish the block if we started it, or we must have a valid solution.
        # Given the speed, we should finish.
        
        if len(block) == k:
            block.sort()
            # Check if this block is already in blocks?
            # Greedy usually generates unique blocks, but let's be safe.
            if block not in blocks:
                blocks.append(block)
                update_uncovered(block)
            else:
                # If duplicate, we still need to update uncovered?
                # No, if it's a duplicate, its sets are already covered.
                # But we might have missed some? No, update_uncovered marks them.
                pass
        else:
            # Incomplete block due to time limit.
            # We should try to complete it quickly with random points?
            # Or just add it if it covers anything?
            # The interface requires k-element blocks.
            # So we MUST pad it.
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            if block not in blocks:
                blocks.append(block)
                update_uncovered(block)

    # Local Search Phase: Iterative Deletion
    # We can try to remove blocks that are redundant.
    # A block is redundant if all its t-subsets are covered by other blocks.
    # This is hard to check efficiently without rebuilding the coverage map.
    # Given time constraints, we might skip this or do a simple version.
    
    # Let's do a simple check:
    # If we have time, try to remove blocks.
    if time.time() < deadline - 2.0:
        # Rebuild coverage count from scratch to check redundancy?
        # Or just try to remove one block at a time and check if any set becomes uncovered.
        # This is O(N_blocks * N_t_sets) which is 300 * 11628 ~ 3.5M ops. Fast.
        
        i = 0
        while i < len(blocks) and time.time() < deadline - 2.0:
            block_to_test = blocks[i]
            # Temporarily remove
            blocks.pop(i)
            
            # Check if any t-set is now uncovered
            # We need to know which sets were ONLY covered by this block.
            # This is complex.
            # Alternative: Just re-verify the whole covering? Too slow.
            
            # Let's skip complex local search to avoid bugs.
            # The greedy is the main improvement.
            
            # Put it back
            blocks.insert(i, block_to_test)
            i += 1

    return blocks

# To make calc_gain robust for |B| < t-1, I will update the logic in the final code.
# The logic above had a flaw. I will fix it in the final version.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_sets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    uncovered = np.ones(n_t_sets, dtype=bool)
    
    # Precompute for each point x, the list of indices of t-sets containing x
    # This helps in the "low block size" case
    point_to_sets = [[] for _ in range(v)]
    for i, s in enumerate(t_subsets):
        for p in s:
            point_to_sets[p].append(i)
            
    # Also, for fast subset checking, we can use a set of tuples for uncovered?
    # No, boolean array is faster for updates.
    # For gain calculation when |B| < t-1:
    # We want x such that it is in many uncovered sets that contain B.
    # We can iterate through point_to_sets[x] and check if B is subset of t_subsets[idx].
    
    blocks = []
    
    def update_uncovered(block):
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            uncovered[idx] = False

    def calc_gain(block, x):
        b_len = len(block)
        if b_len >= t - 1:
            # Standard case: T \ {x} \subseteq B
            gain = 0
            for R in itertools.combinations(block, t - 1):
                T = tuple(sorted(R + (x,)))
                idx = subset_to_idx[T]
                if uncovered[idx]:
                    gain += 1
            return gain
        else:
            # Low block size case:
            # Count uncovered T such that B \subseteq T and x \in T
            gain = 0
            # Iterate through sets containing x
            for idx in point_to_sets[x]:
                if not uncovered[idx]:
                    continue
                T = t_subsets[idx]
                # Check if B is subset of T
                # B is small, T is size t
                # Convert B to set for fast lookup? Or just check membership.
                # Since B is small, checking membership is fast.
                # But T is a tuple.
                # We can check if all(p in T for p in block)
                # Optimization: Precompute a bitmask for each T?
                # v=19, so bitmask fits in int.
                # Let's use bitmasks for subset checks.
                
                # Actually, for v=19, checking membership in a tuple of size 5 is very fast.
                if all(p in T for p in block):
                    gain += 1
            return gain

    while True:
        if time.time() > deadline - 5.0:
            break
        
        unc_idx = np.nonzero(uncovered)[0]
        if len(unc_idx) == 0:
            break
        
        first_unc_set = t_subsets[unc_idx[0]]
        block = [first_unc_set[0]]
        
        while len(block) < k:
            if time.time() > deadline - 5.0:
                # Emergency exit
                # Pad block with random points to make it valid size k
                # This might create a block that covers nothing new, but it's valid.
                while len(block) < k:
                    for x in range(v):
                        if x not in block:
                            block.append(x)
                            break
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    update_uncovered(block)
                break
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                g = calc_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
            
            block.append(best_x)
        
        if len(block) == k:
            block.sort()
            if block not in blocks:
                blocks.append(block)
                update_uncovered(block)
        else:
            # This case is handled by the emergency exit above, 
            # but if we exited normally, block should be size k.
            # If we didn't exit normally, we finished the loop.
            block.sort()
            if block not in blocks:
                blocks.append(block)
                update_uncovered(block)

    return blocks
