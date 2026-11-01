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

    # 2. Greedy Set Cover (for initial solution)
    uncovered = [True] * num_t_subsets
    gain = [0] * num_k_subsets
    
    for b_idx in range(num_k_subsets):
        gain[b_idx] = len(block_t_indices[b_idx])
        
    selected_blocks = []
    
    while True:
        max_gain = max(gain)
        if max_gain == 0:
            break
        best_block = gain.index(max_gain)
        selected_blocks.append(best_block)
        
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    # Initial best solution
    best_blocks = list(selected_blocks)
    best_size = len(best_blocks)
    
    # 3. Branch and Bound
    start_time = time.time()
    time_limit = 110.0
    
    # Helper to check if a block covers any uncovered sets
    # We use a set of uncovered indices for efficiency in B&B
    
    def bnb(uncovered_set, current_depth, current_blocks):
        nonlocal best_blocks, best_size
        if time.time() - start_time > time_limit:
            return
            
        if not uncovered_set:
            if current_depth < best_size:
                best_size = current_depth
                best_blocks = list(current_blocks)
            return
            
        # Lower Bound
        # Max t-subsets a single block can cover is C(k,t)
        max_cover = len(block_t_indices[0])
        lb = (len(uncovered_set) + max_cover - 1) // max_cover
        
        if current_depth + lb >= best_size:
            return
            
        # Pick a t-subset with fewest candidates to reduce branching
        # To speed up, we pick a random one from the first few?
        # Or the one with min len(t_subset_blocks)?
        # We can't easily get min degree without iterating.
        # Let's pick the first one for simplicity, or randomize.
        t_idx = random.choice(list(uncovered_set))
        
        candidates = t_subset_blocks[t_idx]
        
        # Sort by gain (coverage of uncovered_set)
        # Optimization: Only consider blocks that cover at least 1 other uncovered set?
        # No, they must cover t_idx.
        
        def get_gain(b_idx):
            # Count how many t in block_t_indices[b_idx] are in uncovered_set
            # Using set intersection size
            return len(uncovered_set.intersection(block_t_indices[b_idx]))
            
        # Sort candidates
        # To save time, we can limit the sorting to a subset or use a fast heuristic
        candidates.sort(key=get_gain, reverse=True)
        
        # Limit branching factor
        # If we branch on all 455, it's too slow.
        # Top 5-10 should be sufficient for finding good covers.
        top_candidates = candidates[:10]
        
        for b_idx in top_candidates:
            # Remove covered sets
            new_uncovered = uncovered_set.difference(block_t_indices[b_idx])
            current_blocks.append(b_idx)
            bnb(new_uncovered, current_depth + 1, current_blocks)
            current_blocks.pop()
            
    # Run B&B
    # Start with all uncovered
    initial_uncovered = set(range(num_t_subsets))
    bnb(initial_uncovered, 0, [])
    
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
