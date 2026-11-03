import itertools
import random
import time
import heapq

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin under 120s limit

    # 1. Precomputation
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    block_masks = [0] * num_blocks
    block_tsubsets = []
    for i, block in enumerate(all_blocks):
        mask = 0
        tsubs = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
            tsubs.append(idx)
        block_masks[i] = mask
        block_tsubsets.append(tsubs)

    best_solution = None
    best_size = float('inf')
    
    # 2. Multiple Restart Randomized Greedy + Local Search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Randomized Greedy
        uncovered_mask = (1 << num_subsets) - 1
        selected_blocks = []
        counts = [0] * num_subsets
        
        # Increase randomness over time
        top_k = 1 + (iteration % 20)
        
        while uncovered_mask:
            if time.time() > deadline - 10.0: # Leave 10s for local search and potential next iteration
                break
            
            # Calculate gains
            # Optimization: Only consider blocks that might have gain
            # We can't easily filter without checking.
            
            candidates = []
            best_gain = 0
            
            for i in range(num_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > 0:
                    candidates.append((gain, i))
                    if gain > best_gain:
                        best_gain = gain
            
            if not candidates:
                break
                
            # Pick from top K
            if top_k == 1:
                # Greedy: pick max
                # Find max efficiently? candidates is already built.
                # We can just use max(candidates, key=lambda x: x[0])
                chosen_idx = max(candidates, key=lambda x: x[0])[1]
            else:
                # Randomized: pick from top K
                # Use heapq.nlargest for efficiency if K is small
                top_candidates = heapq.nlargest(top_k, candidates, key=lambda x: x[0])
                chosen_idx = random.choice(top_candidates)[1]
            
            # Select the block
            selected_blocks.append(chosen_idx)
            mask = block_masks[chosen_idx]
            
            # Update counts
            for ts_idx in block_tsubsets[chosen_idx]:
                counts[ts_idx] += 1
                
            # Update uncovered mask
            uncovered_mask &= ~mask
            
            if best_gain == 0:
                break

        # Local Search: Remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for b_idx in selected_blocks:
                redundant = True
                for ts_idx in block_tsubsets[b_idx]:
                    if counts[ts_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for ts_idx in block_tsubsets[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected_blocks = new_selected

        # Check if this solution is better
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = [list(all_blocks[i]) for i in selected_blocks]

    if best_solution is None:
        return []
        
    return best_solution
