import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    best_blocks = None
    start_time = time.time()
    deadline = start_time + 120.0
    
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        uncovered = np.ones(n_subsets, dtype=bool)
        cover_count = np.zeros(n_subsets, dtype=int)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered subset to start a new block
            uncovered_indices = np.where(uncovered)[0]
            idx = random.choice(uncovered_indices)
            start_subset = all_t_subsets[idx]
            
            # Initialize block with this subset
            block = list(start_subset)
            
            # Add points to complete the block
            while len(block) < k:
                best_gain = -1
                candidates = []
                
                # Calculate gain for each candidate point
                for p in range(v):
                    if p in block:
                        continue
                    
                    gain = 0
                    # Check all (t-1) subsets of current block
                    for comb in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(comb + (p,)))
                        new_idx = subset_to_idx[new_subset]
                        if uncovered[new_idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [p]
                    elif gain == best_gain:
                        candidates.append(p)
                
                # Pick randomly from best candidates to add diversity
                if candidates:
                    block.append(random.choice(candidates))
                else:
                    # Fallback
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        block.append(random.choice(remaining))
                    else:
                        break # Should not happen if k <= v
            
            # Sort block for consistency
            block = tuple(sorted(block))
            blocks.append(block)
            
            # Update coverage
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                cover_count[idx] += 1
                if cover_count[idx] == 1:
                    uncovered[idx] = False
        
        # 2. Redundancy Removal
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    idx = subset_to_idx[sub]
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for sub in itertools.combinations(b, t):
                        cover_count[subset_to_idx[sub]] -= 1
                    blocks.pop(i)
                    changed = True
        
        # 3. Update Best
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = [list(b) for b in blocks]
        
        # Check time again before next iteration
        if time.time() > deadline - 5:
            break

    if best_blocks is None:
        # Fallback: return empty or a single block if something went wrong
        return []
        
    return best_blocks
