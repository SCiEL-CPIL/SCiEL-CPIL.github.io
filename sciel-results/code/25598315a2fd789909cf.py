import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using Simulated Annealing on the set of blocks.
    This is a metaheuristic approach distinct from the greedy constructive method.
    """
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and their coverage indices
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # Map each k-subset to the indices of the t-subsets it covers
    # Using a list of lists for flexibility, but we will convert to numpy for fast ops if needed.
    # For SA, we need to update counts quickly.
    block_cover_indices = []
    for block in all_k_subsets:
        t_subsets_in_block = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in t_subsets_in_block]
        block_cover_indices.append(indices)
        
    # Convert to numpy array for faster access? 
    # Accessing a row in a list of lists is fast enough in Python if we don't do it millions of times per step.
    # However, updating counts: for i in block_cover_indices[block_id]: counts[i] += 1
    # This is C(k,t) operations. C(7,3)=35. This is very fast.
    
    # 1. Generate an initial valid solution using Greedy Set Cover
    # This ensures we start in a "good" region of the search space.
    
    def greedy_initial():
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_found = []
        # We need to find the best block quickly.
        # To avoid O(N_k * C(k,t)) per step in pure Python which might be slow for 100 restarts,
        # we will use a simplified greedy: pick a random uncovered t-subset, then pick a random block covering it?
        # No, that's not greedy.
        # Let's use the vectorized greedy from the original but run it just ONCE or a few times to get a start.
        # Actually, let's just run the vectorized greedy once to get a baseline.
        
        cover_matrix = np.array(block_cover_indices, dtype=np.int32)
        
        while True:
            if not np.any(uncovered):
                break
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[random.randint(0, len(candidates)-1)]
            blocks_found.append(chosen_idx)
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks_found

    # Get initial solution
    random.seed(42)
    initial_blocks = greedy_initial()
    
    # State for SA
    current_blocks = list(initial_blocks)
    # counts[i] = number of blocks covering t-subset i
    counts = np.zeros(n_t_subsets, dtype=np.int32)
    for b_idx in current_blocks:
        for t_idx in block_cover_indices[b_idx]:
            counts[t_idx] += 1
            
    def get_uncovered_count(counts_arr):
        return np.sum(counts_arr == 0)
        
    current_uncovered = get_uncovered_count(counts)
    current_cost = len(current_blocks) + 1000 * current_uncovered # High penalty for invalid
    
    best_blocks = list(current_blocks) if current_uncovered == 0 else []
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 115
    
    # SA Parameters
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    while T > T_min:
        if time.time() - start_time > timeout:
            break
            
        # Generate a neighbor
        # Move types: Add, Remove, Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_blocks = list(current_blocks)
        new_counts = counts.copy()
        
        if move_type == 'add':
            # Pick a random block not in current_blocks
            # To avoid O(N) check, we can just pick a random block and check membership?
            # Membership in list is O(N). N~76. This is fine.
            # But we want to avoid adding duplicates.
            # Let's pick a random index from 0 to n_k_subsets-1
            candidate = random.randint(0, n_k_subsets - 1)
            if candidate not in new_blocks:
                new_blocks.append(candidate)
                for t_idx in block_cover_indices[candidate]:
                    new_counts[t_idx] += 1
            else:
                # If duplicate, try a few times or just skip move
                # To save time, just skip
                T *= cooling
                continue
                
        elif move_type == 'remove':
            if not new_blocks:
                T *= cooling
                continue
            # Pick a random block to remove
            rem_idx = random.randint(0, len(new_blocks) - 1)
            candidate = new_blocks[rem_idx]
            new_blocks.pop(rem_idx)
            for t_idx in block_cover_indices[candidate]:
                new_counts[t_idx] -= 1
                
        elif move_type == 'swap':
            if not new_blocks:
                T *= cooling
                continue
            # Pick a block to remove and a block to add
            rem_idx = random.randint(0, len(new_blocks) - 1)
            rem_candidate = new_blocks[rem_idx]
            
            # Pick a random block to add
            add_candidate = random.randint(0, n_k_subsets - 1)
            if add_candidate == rem_candidate:
                T *= cooling
                continue
            if add_candidate in new_blocks:
                # Avoid swapping with existing block (effectively no-op or duplicate)
                T *= cooling
                continue
                
            new_blocks[rem_idx] = add_candidate
            # Update counts
            for t_idx in block_cover_indices[rem_candidate]:
                new_counts[t_idx] -= 1
            for t_idx in block_cover_indices[add_candidate]:
                new_counts[t_idx] += 1
                
        new_uncovered = get_uncovered_count(new_counts)
        new_cost = len(new_blocks) + 1000 * new_uncovered
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_blocks = new_blocks
            counts = new_counts
            current_uncovered = new_uncovered
            current_cost = new_cost
            
            if new_uncovered == 0 and new_cost < best_cost:
                best_cost = new_cost
                best_blocks = list(new_blocks)
                
        T *= cooling
        
    # Convert indices back to blocks
    if not best_blocks:
        # Fallback to initial if SA failed to find valid one (unlikely given start)
        best_blocks = initial_blocks
        
    return [all_k_subsets[i] for i in best_blocks]
