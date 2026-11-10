import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    # Precompute t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # Map each k-subset to the indices of t-subsets it covers
    block_t_indices = []
    for k_sub in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_subset_to_idx[t_sub])
        block_t_indices.append(t_indices)
        
    # Map t-subset to list of blocks covering it (for initialization)
    t_to_blocks = [[] for _ in range(num_t)]
    for j, t_indices in enumerate(block_t_indices):
        for i in t_indices:
            t_to_blocks[i].append(j)
            
    # --- Phase 1: Randomized Greedy Initialization ---
    selected = np.zeros(num_k, dtype=bool)
    cover_count = np.zeros(num_t, dtype=np.int32)
    uncovered_indices = list(range(num_t))
    uncovered_set = set(uncovered_indices)
    num_selected = 0
    
    while uncovered_indices:
        # Pick a random uncovered t-subset
        idx = random.randint(0, len(uncovered_indices) - 1)
        ti = uncovered_indices.pop(idx)
        uncovered_set.remove(ti)
        
        # Pick a random block covering this t-subset
        # To improve quality, we could pick the block that covers the most *other* uncovered subsets.
        # But that's expensive. Let's stick to random for now, or simple heuristic.
        # Heuristic: Pick the block that covers the most uncovered t-subsets among its neighbors?
        # Let's just pick random for speed.
        
        candidates = t_to_blocks[ti]
        # To make it slightly smarter, pick the candidate with the highest "potential"
        # But for now, random is O(1).
        
        j = random.choice(candidates)
        
        if not selected[j]:
            selected[j] = True
            num_selected += 1
            # Update cover counts and uncovered status
            for t_idx in block_t_indices[j]:
                if cover_count[t_idx] == 0:
                    # This t-subset becomes covered
                    if t_idx in uncovered_set:
                        # Remove from uncovered list efficiently?
                        # Set removal is O(1), but list pop is O(N).
                        # We'll just mark it and filter later or use a boolean array.
                        pass
                cover_count[t_idx] += 1
                
            # Re-filter uncovered_indices is O(N_t). 
            # Better to maintain a boolean array `is_uncovered`.
            
    # The above loop is inefficient for list management.
    # Let's use a boolean array `is_uncovered`.
    
    selected = np.zeros(num_k, dtype=bool)
    cover_count = np.zeros(num_t, dtype=np.int32)
    is_uncovered = np.ones(num_t, dtype=bool)
    num_uncovered = num_t
    num_selected = 0
    
    while num_uncovered > 0:
        # Find an uncovered t-subset. 
        # We can iterate through is_uncovered to find one.
        # To speed up, we can pick a random index and check.
        found = False
        for _ in range(100): # Try random 100 times
            ti = random.randint(0, num_t - 1)
            if is_uncovered[ti]:
                found = True
                break
        if not found:
            # Fallback: linear scan
            for ti in range(num_t):
                if is_uncovered[ti]:
                    found = True
                    break
            if not found:
                break
                
        # Pick a block covering ti
        # Heuristic: Pick the block that covers the most uncovered t-subsets.
        best_j = -1
        best_count = -1
        candidates = t_to_blocks[ti]
        # To avoid O(N_k) scan, we only check candidates for this ti.
        # But we want a block that covers *many* uncovered ones.
        # We can sample a few candidates.
        sample_size = min(10, len(candidates))
        sampled = random.sample(candidates, sample_size)
        
        for j in sampled:
            if selected[j]:
                continue
            count = 0
            for t_idx in block_t_indices[j]:
                if is_uncovered[t_idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_j = j
                
        if best_j == -1:
            # No candidate found? Should not happen.
            break
            
        selected[best_j] = True
        num_selected += 1
        
        for t_idx in block_t_indices[best_j]:
            if cover_count[t_idx] == 0:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    num_uncovered -= 1
            cover_count[t_idx] += 1
            
    # --- Phase 2: Simulated Annealing ---
    # Objective: Minimize num_selected.
    # We allow temporary increases in num_uncovered with penalty.
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    steps = 50000 # Adjust based on time
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    best_num_selected = num_selected
    best_selected = selected.copy()
    best_cover_count = cover_count.copy()
    best_is_uncovered = is_uncovered.copy()
    best_num_uncovered = num_uncovered
    
    current_num_selected = num_selected
    current_cover_count = cover_count.copy()
    current_is_uncovered = is_uncovered.copy()
    current_num_uncovered = num_uncovered
    current_selected = selected.copy()
    
    while T > min_T and time.time() - start_time < timeout:
        # Pick a random block to flip
        j = random.randint(0, num_k - 1)
        
        if current_selected[j]:
            # Try to remove
            # Check if any t-subset in block j has cover_count == 1
            delta_uncovered = 0
            for t_idx in block_t_indices[j]:
                if current_cover_count[t_idx] == 1:
                    delta_uncovered += 1
                    
            delta_selected = -1
            delta_E = 1000 * delta_uncovered + delta_selected
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                current_selected[j] = False
                for t_idx in block_t_indices[j]:
                    current_cover_count[t_idx] -= 1
                    if current_cover_count[t_idx] == 0:
                        if current_is_uncovered[t_idx]:
                            pass # Already uncovered
                        else:
                            current_is_uncovered[t_idx] = True
                            current_num_uncovered += 1
                    # If it was 1 and becomes 0, it becomes uncovered
                # Update num_selected
                current_num_selected += delta_selected
                
                # Update best if valid and better
                if current_num_uncovered == 0 and current_num_selected < best_num_selected:
                    best_num_selected = current_num_selected
                    best_selected = current_selected.copy()
                    best_cover_count = current_cover_count.copy()
                    best_is_uncovered = current_is_uncovered.copy()
                    best_num_uncovered = 0
                    
        else:
            # Try to add
            # Check if any t-subset in block j has cover_count == 0
            delta_uncovered = 0
            for t_idx in block_t_indices[j]:
                if current_cover_count[t_idx] == 0:
                    delta_uncovered -= 1 # Reduces uncovered count
                    
            delta_selected = 1
            delta_E = 1000 * delta_uncovered + delta_selected
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                current_selected[j] = True
                for t_idx in block_t_indices[j]:
                    if current_cover_count[t_idx] == 0:
                        current_is_uncovered[t_idx] = False
                        current_num_uncovered -= 1
                    current_cover_count[t_idx] += 1
                    
                current_num_selected += delta_selected
                
                if current_num_uncovered == 0 and current_num_selected < best_num_selected:
                    best_num_selected = current_num_selected
                    best_selected = current_selected.copy()
                    best_cover_count = current_cover_count.copy()
                    best_is_uncovered = current_is_uncovered.copy()
                    best_num_uncovered = 0
                    
        T *= cooling
        
    # Convert best_selected to list of blocks
    blocks = []
    for j in range(num_k):
        if best_selected[j]:
            blocks.append(list(all_k_subsets[j]))
            
    return blocks
