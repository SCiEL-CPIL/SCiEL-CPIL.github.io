import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each point x, the list of t-subsets containing x?
    # Or just use the index lookup directly.
    
    uncovered = set(range(num_t_subs))
    blocks = []
    
    # Greedy Construction
    while uncovered and time.time() < deadline:
        # Pick a seed t-subset. Using min index is deterministic.
        # Randomizing might help, but let's stick to deterministic first.
        seed_idx = min(uncovered)
        seed_sub = all_t_subs[seed_idx]
        
        block = list(seed_sub)
        
        while len(block) < k:
            best_gain = -1
            best_point = -1
            # Candidates are points not in block
            # To optimize, we can precompute combinations of block of size t-1
            # But block size is small, so generating them on the fly is fine.
            
            # Optimization: Precompute the (t-1)-subsets of the current block
            # and map them to the potential t-subsets they form with each candidate x.
            # Actually, just iterating candidates is fine for v=19.
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms an uncovered t-subset
                # Note: The (t-1)-subset must not contain x (which is true by construction)
                for sub in itertools.combinations(block, t - 1):
                    # Create the t-subset
                    # Since x is not in block, sub + (x,) is a valid set
                    # We need to sort it to match the canonical form in t_sub_to_idx
                    # Wait, itertools.combinations produces sorted tuples if input is sorted.
                    # block is a list, but we should ensure we treat it as a set for combination generation?
                    # No, combinations works on the sequence. If block is [0, 2, 5], combinations(block, 2) gives (0,2), (0,5), (2,5).
                    # Adding x: if x=1, (0,1,2) is sorted. If x=3, (0,2,3) is sorted.
                    # But if block=[0, 2] and x=1, sub=(0,2), sub+(x,)=(0,2,1) which is NOT sorted.
                    # So we must sort.
                    new_sub = tuple(sorted(sub + (x,)))
                    if new_sub in t_sub_to_idx:
                        idx = t_sub_to_idx[new_sub]
                        if idx in uncovered:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    if gain == len(uncovered): # Theoretical max, unlikely
                        break
            
            if best_point == -1:
                break # Should not happen if v > k
                
            block.append(best_point)
            block.sort()
            
        # Add block
        blocks.append(tuple(block))
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if idx in uncovered:
                uncovered.remove(idx)
                
    # If we didn't finish greedy, we have an invalid covering.
    # But greedy always finishes quickly for these sizes.
    # If uncovered is not empty, we need to fix it?
    # The greedy algorithm as described *always* covers everything because it explicitly targets uncovered subsets.
    # The only risk is timeout.
    
    if uncovered:
        # Emergency fallback: brute force add blocks for remaining uncovered?
        # Or just return what we have (invalid). Better to try to finish.
        # But 110s is a lot. Greedy should be done in < 1s.
        pass

    # Redundancy Removal
    # Initialize cover counts
    cover_count = np.zeros(num_t_subs, dtype=np.int32)
    block_t_subs = []
    for b in blocks:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)
        for idx in subs:
            cover_count[idx] += 1
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            subs = block_t_subs[i]
            
            # Check if all t-subsets in this block are covered by at least one other block
            redundant = True
            for idx in subs:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                # Decrement cover counts
                for idx in subs:
                    cover_count[idx] -= 1
                
                # Remove from lists
                blocks.pop(i)
                block_t_subs.pop(i)
                changed = True
                break # Restart loop because indices changed
    
    # Convert back to list of lists
    return [list(b) for b in blocks]
