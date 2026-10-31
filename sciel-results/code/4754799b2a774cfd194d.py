import itertools
import random
import time
import sys

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # Helper: Run one greedy pass with randomization
    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = [True] * num_t_subsets
        gain = [len(block_t_indices[b]) for b in range(num_k_subsets)]
        
        selected_blocks = []
        
        # Use a max-heap or just scan? Scanning is O(N). N=50k.
        # 150 iterations * 50k = 7.5M ops. Fast in Python? Maybe 1-2 seconds.
        # With randomization, we might do 10-20 restarts. 20 * 2s = 40s. OK.
        
        while True:
            # Find max gain with random tie-breaking
            max_gain = -1
            best_block = -1
            # To speed up tie-breaking, we can collect all maxes and pick random
            # Or just add noise.
            # Let's scan and keep track of max, and if equal, pick randomly.
            # This requires storing the list of best candidates.
            
            candidates = []
            current_max = -1
            
            for b_idx in range(num_k_subsets):
                g = gain[b_idx]
                if g > current_max:
                    current_max = g
                    candidates = [b_idx]
                elif g == current_max and g > 0:
                    candidates.append(b_idx)
            
            if current_max == 0:
                break
                
            # Randomly pick from candidates
            best_block = random.choice(candidates)
            selected_blocks.append(best_block)
            
            # Update gains
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks

    # Helper: Minimalization
    def minimize(blocks):
        if not blocks:
            return []
            
        # Use a set for O(1) lookup
        active_set = set(blocks)
        cover_count = [0] * num_t_subsets
        
        for b_idx in blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        # Multiple passes of deletion
        # Shuffle order to avoid bias
        shuffled_blocks = list(blocks)
        random.shuffle(shuffled_blocks)
        
        removed = False
        # Repeat passes until no more removals
        while removed:
            removed = False
            for b_idx in shuffled_blocks:
                if b_idx not in active_set:
                    continue
                    
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    active_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    removed = True
                    
        return list(active_set)

    # Main Loop: Randomized Restarts
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead/safety
    
    best_solution = None
    best_size = float('inf')
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        solution = run_greedy(iteration)
        minimized = minimize(solution)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_solution = minimized
            
        # Optional: Early stop if we hit the lower bound? 
        # Lower bound is 131. If we hit 131, we can stop.
        if best_size == 131:
            break
            
    if best_solution is None:
        # Fallback to deterministic greedy if no time left for loop
        solution = run_greedy(0)
        best_solution = minimize(solution)

    return [list(all_k_subsets[b_idx]) for b_idx in best_solution]
