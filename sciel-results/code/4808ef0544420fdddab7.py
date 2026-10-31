import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    best_solution = None
    best_size = float('inf')
    start_time = time.time()
    
    # Run multiple randomized greedy passes
    # We estimate ~0.5s per pass. 120s allows ~200 passes? 
    # But deletion can be slow. Let's be conservative: 50 passes.
    
    for run in range(50):
        if time.time() - start_time > 110:
            break
            
        # Randomized Greedy
        shuffled_blocks = list(range(num_k_subsets))
        random.shuffle(shuffled_blocks)
        
        uncovered = [True] * num_t_subsets
        gain = [0] * num_k_subsets
        
        # Initialize gain
        for b_idx in range(num_k_subsets):
            gain[b_idx] = len(block_t_indices[b_idx])
            
        selected_blocks = []
        
        while True:
            max_gain = max(gain)
            if max_gain == 0:
                break
                
            # Find first block in shuffled_blocks with gain == max_gain
            best_block = -1
            for b_idx in shuffled_blocks:
                if gain[b_idx] == max_gain:
                    best_block = b_idx
                    break
            
            if best_block == -1:
                best_block = gain.index(max_gain)
                
            selected_blocks.append(best_block)
            
            # Update gains
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        # Minimalization: Iterative Deletion
        cover_count = [0] * num_t_subsets
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        active_blocks = set(selected_blocks)
        
        # Repeat deletion passes until no change
        changed = True
        while changed:
            changed = False
            current_list = list(active_blocks)
            random.shuffle(current_list)
            
            for b_idx in current_list:
                if b_idx not in active_blocks:
                    continue
                    
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    active_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
                    
        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_solution = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
            
    return best_solution
