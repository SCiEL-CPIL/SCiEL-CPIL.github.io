import itertools
import random
import math
import time

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Precompute all k-subsets and their t-subset indices
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # block_t_indices[i] is a list of t-subset indices covered by k-subset i
    block_t_indices = []
    for k_sub in k_subsets:
        indices = [t_subset_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)]
        block_t_indices.append(indices)
        
    # Precompute block_masks for initial greedy
    block_masks = [0] * n_k
    for i, indices in enumerate(block_t_indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        block_masks[i] = mask

    # 1. Initial Solution: Greedy Set Cover
    uncovered_mask = (1 << n_t) - 1
    greedy_blocks = []
    in_greedy = [False] * n_k
    
    while uncovered_mask:
        best_idx = -1
        max_cov = 0
        for i in range(n_k):
            if in_greedy[i]: continue
            cov = (uncovered_mask & block_masks[i]).bit_count()
            if cov > max_cov:
                max_cov = cov
                best_idx = i
        if max_cov == 0: break
        greedy_blocks.append(best_idx)
        in_greedy[best_idx] = True
        uncovered_mask &= ~block_masks[best_idx]
        
    best_solution = list(greedy_blocks)
    best_cost = len(best_solution)
    
    # 2. Simulated Annealing
    current_blocks = list(greedy_blocks)
    in_current = [False] * n_k
    for b in current_blocks:
        in_current[b] = True
        
    cover_count = [0] * n_t
    for b_idx in current_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    current_cost = len(current_blocks) + 100 * uncovered_count
    
    T = 5.0
    T_min = 0.01
    alpha = 0.995
    start_time = time.time()
    time_limit = 110.0
    
    while T > T_min and (time.time() - start_time) < time_limit:
        if random.random() < 0.5:
            # Remove a random block
            if not current_blocks: continue
            b_idx = random.choice(current_blocks)
            
            # Calculate delta
            delta_uncovered = 0
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 1:
                    delta_uncovered += 1
                cover_count[t_idx] -= 1
            
            new_blocks = len(current_blocks) - 1
            new_uncovered = uncovered_count + delta_uncovered
            new_cost = new_blocks + 100 * new_uncovered
            
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                # Accept
                # Remove b_idx from current_blocks
                # Find index
                pos = current_blocks.index(b_idx)
                current_blocks[pos] = current_blocks[-1]
                current_blocks.pop()
                in_current[b_idx] = False
                uncovered_count = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_blocks)
            else:
                # Revert
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] += 1
                    
        else:
            # Add a random block
            b_idx = random.randint(0, n_k - 1)
            if in_current[b_idx]:
                continue
                
            delta_uncovered = 0
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 0:
                    delta_uncovered -= 1
                cover_count[t_idx] += 1
            
            new_blocks = len(current_blocks) + 1
            new_uncovered = uncovered_count + delta_uncovered
            new_cost = new_blocks + 100 * new_uncovered
            
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                # Accept
                current_blocks.append(b_idx)
                in_current[b_idx] = True
                uncovered_count = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_blocks)
            else:
                # Revert
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
        
        T *= alpha
        
    # Ensure we return a valid solution
    # Check best_solution validity
    final_cover_count = [0] * n_t
    for b_idx in best_solution:
        for t_idx in block_t_indices[b_idx]:
            final_cover_count[t_idx] += 1
            
    if any(c == 0 for c in final_cover_count):
        # Fallback to greedy
        return [list(k_subsets[i]) for i in greedy_blocks]
        
    return [list(k_subsets[i]) for i in best_solution]
