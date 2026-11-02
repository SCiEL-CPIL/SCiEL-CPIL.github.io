import numpy as np
import itertools
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_subs)
    
    # Map each subset to its index
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    
    # Convert to numpy array for faster access if needed, 
    # but we mostly use the index mapping.
    # subs_array = np.array(all_subs)
    
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []
    
    # Precompute for each point x, the list of indices of t-subsets containing x?
    # Not strictly necessary for the gain calculation logic derived above,
    # but might help if we change strategy.
    # The current strategy: for candidate x, find subsets {x} U C where C in C(B, t-1).
    
    # To speed up "index of subset", we use the dictionary sub_to_idx.
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    while not covered.all():
        if time.time() > deadline:
            break
            
        # Find first uncovered subset
        # np.argmax returns the index of the first True in ~covered
        # If all are covered, argmax returns 0, but the loop condition prevents this.
        idx = np.argmax(~covered)
        
        block = list(all_subs[idx])
        
        while len(block) < k:
            best_gain = -1
            best_x = -1
            
            # Candidates are points not in block
            # We can iterate over range(v)
            for x in range(v):
                if x in block:
                    continue
                
                # Calculate gain: number of uncovered t-subsets S 
                # such that x in S and |S \ {x} & block| == t-1
                # This is equivalent to S = {x} U C where C is a (t-1)-subset of block.
                
                # Generate (t-1)-subsets of block
                # t is 4, so t-1 is 3.
                # block size is at least t (4) and at most k-1 (6).
                
                gain = 0
                # itertools.combinations(block, t-1)
                for C in itertools.combinations(block, t - 1):
                    # C is a tuple of t-1 points
                    # The subset is C + (x,)
                    # We need to sort it to match the keys in sub_to_idx
                    # all_subs are generated in lexicographic order, 
                    # so they are sorted tuples.
                    S = tuple(sorted(C + (x,)))
                    s_idx = sub_to_idx[S]
                    if not covered[s_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Should not happen if k <= v
                break
                
            block.append(best_x)
            block.sort() # Keep block sorted for consistency, though not strictly needed for logic
            
            # Early exit if we found a point with 0 gain? 
            # No, we must fill the block to k points.
            # However, if best_gain is 0, adding x covers nothing new.
            # We still add it to complete the block size?
            # The problem says "k-element blocks".
            # So we must add k points.
            
        # Finalize block
        block.sort()
        blocks.append(block)
        
        # Update covered mask
        # All t-subsets of block are now covered
        for S in itertools.combinations(block, t):
            s_idx = sub_to_idx[S]
            covered[s_idx] = True
            
    return blocks
