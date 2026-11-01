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
    
    # Map block index to list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map t-subset index to list of block indices covering it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    def greedy_cover(seed_val):
        random.seed(seed_val)
        uncovered = [True] * num_t_subsets
        gain = [len(block_t_indices[b]) for b in range(num_k_subsets)]
        selected = []
        
        for _ in range(num_t_subsets):
            if not any(uncovered):
                break
                
            max_gain = -1
            best_candidates = []
            
            for b in range(num_k_subsets):
                if gain[b] > max_gain:
                    max_gain = gain[b]
                    best_candidates = [b]
                elif gain[b] == max_gain and max_gain > 0:
                    best_candidates.append(b)
            
            if max_gain == 0:
                break
            
            best_block = random.choice(best_candidates)
            selected.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected

    def minimize_solution(selected_blocks):
        active = set(selected_blocks)
        
        cover_count = [0] * num_t_subsets
        for b_idx in active:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        # Repeated deletion passes
        # We shuffle to ensure we don't always remove the "same" type first
        for _ in range(5): # 5 passes of deletion
            blocks_list = list(active)
            random.shuffle(blocks_list)
            changed = False
            for b_idx in blocks_list:
                if b_idx not in active:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
            if not changed:
                break
                
        return list(active)

    best_solution = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110
    
    # Run multiple greedy instances
    # Estimate: 1 greedy run ~ 0.5s? 
    # Let's try 100 runs? 
    # If 1 run is 0.1s, 100 runs is 10s.
    # If 1 run is 0.5s, 100 runs is 50s.
    # Let's cap at 50 runs to be safe.
    
    for i in range(50):
        if time.time() - start_time > time_limit:
            break
            
        sol = greedy_cover(i)
        sol = minimize_solution(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
    if best_solution is None:
        best_solution = greedy_cover(0)
        best_solution = minimize_solution(best_solution)
        
    return [list(all_k_subsets[b_idx]) for b_idx in best_solution]
