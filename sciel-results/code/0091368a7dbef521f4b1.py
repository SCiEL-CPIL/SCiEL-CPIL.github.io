import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    t_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Precompute masks for each k-subset
    # mask[i] is an integer with bits set for t-subsets contained in k_subs[i]
    masks = [0] * n_k
    for i, block in enumerate(k_subs):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << t_to_idx[t_sub])
        masks[i] = mask
        
    best_blocks = None
    best_count = float('inf')
    
    # 2. Multiple Greedy Runs
    # We can try a few times with random tie-breaking
    for run in range(5):
        if time.time() > deadline - 20: # Reserve 20s for refinement and overhead
            break
            
        uncovered = (1 << n_t) - 1
        blocks = []
        
        # Shuffle block indices to break ties randomly?
        # Or just use random.choice among max gainers?
        # Simpler: Iterate in a random order? No, we need to find MAX.
        # We can add a tiny random noise to the gain?
        # Or, when gain == best_gain, we can randomly decide to update best_idx.
        
        while uncovered:
            if time.time() > deadline - 10:
                # Timeout during greedy, discard this run or keep partial?
                # Discard is safer for validity if we don't check coverage.
                # But if we return partial, it's invalid.
                # So we must finish the run or abort.
                # If we abort, we return best_blocks so far.
                break
                
            best_gain = 0
            best_idx = -1
            max_possible = min(uncovered.bit_count(), 56)
            
            # To speed up, we can skip blocks that we know are "bad"?
            # No, just scan.
            for i in range(n_k):
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if best_gain == max_possible:
                        break
                elif gain == best_gain and gain > 0:
                    # Random tie breaking
                    if random.random() < 0.5:
                        best_idx = i
                        if best_gain == max_possible:
                            break
            
            if best_idx == -1:
                # Should not happen if uncovered is not 0
                break
                
            blocks.append(best_idx)
            uncovered &= ~masks[best_idx]
            
        if time.time() > deadline - 10:
            continue # Don't refine if we are out of time
            
        # 3. Refinement: Remove redundant blocks
        # blocks is a list of indices into k_subs
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, idx in enumerate(blocks):
                # Check if mask[idx] is subset of union of others
                # Optimization: If blocks is small, this is fast.
                union_others = 0
                for j, idx_j in enumerate(blocks):
                    if j != i:
                        union_others |= masks[idx_j]
                
                if (masks[idx] & ~union_others) == 0:
                    changed = True
                    continue
                new_blocks.append(idx)
            blocks = new_blocks
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    # 4. Convert indices back to lists of points
    if best_blocks is None:
        # Fallback: return empty or a trivial covering?
        # Must return valid covering.
        # If no run finished, we might have an invalid one.
        # Ensure we return at least one valid covering.
        # The first run should finish.
        pass
        
    return [list(k_subs[i]) for i in best_blocks]
