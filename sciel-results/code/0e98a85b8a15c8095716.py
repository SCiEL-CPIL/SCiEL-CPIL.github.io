import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0
    
    # Precompute t-subsets and their indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    t_sub_to_idx = {s: i for i, s in enumerate(t_subs)}
    
    # Precompute k-subsets (blocks) and their masks
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # block_masks[b] is an integer bitset where bit i is 1 if t_subs[i] is in k_subs[b]
    block_masks = [0] * n_k
    for b_idx, block in enumerate(k_subs):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    all_mask = (1 << n_t) - 1
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    while time.time() - start_time < timeout:
        # Greedy Phase
        uncovered = all_mask
        active_blocks = set(range(n_k))
        solution = []
        
        while uncovered:
            best_gain = -1
            best_block = -1
            # Find block with max gain
            # Optimization: If we have many blocks, iterating all is O(N).
            for b in active_blocks:
                gain = (uncovered & block_masks[b]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    # Reset tie counter
                    tie_count = 1
                elif gain == best_gain and gain > 0:
                    tie_count += 1
                    # Randomly pick one of the ties to add diversity
                    if random.random() < 1.0 / tie_count:
                        best_block = b
                # Early termination if we covered all remaining subsets
                if gain == uncovered.bit_count():
                    break
            
            if best_block == -1:
                break
                
            solution.append(best_block)
            active_blocks.remove(best_block)
            uncovered &= ~block_masks[best_block]
            
        # Reduction Phase
        changed = True
        while changed:
            changed = False
            for i in range(len(solution) - 1, -1, -1):
                b = solution[i]
                union_others = 0
                for j in range(len(solution)):
                    if j != i:
                        union_others |= block_masks[solution[j]]
                
                if (block_masks[b] & ~union_others) == 0:
                    solution.pop(i)
                    changed = True
                    break
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
        if best_size <= 131:
            break

    if best_solution is None:
        return []
        
    final_blocks = [list(k_subs[idx]) for idx in best_solution]
    return final_blocks
