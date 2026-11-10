import itertools
import random


def construct(v, k, t):
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # For each k-subset, compute the list of t-subset indices it covers
    k_to_t_indices = []
    for k_sub in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_subset_to_idx[t_sub])
        k_to_t_indices.append(t_indices)
    
    # Reverse mapping: for each t-subset index, which k-subsets cover it
    t_to_k_indices = [[] for _ in range(num_t)]
    for ki, t_indices in enumerate(k_to_t_indices):
        for ti in t_indices:
            t_to_k_indices[ti].append(ki)
            
    max_cov = len(k_to_t_indices[0]) if num_k > 0 else 0

    def run_greedy(order):
        coverage = [max_cov] * num_k
        uncovered = [True] * num_t
        selected = [False] * num_k
        uncovered_count = num_t
        
        # Initialize buckets
        buckets = [[] for _ in range(max_cov + 1)]
        for ki in order:
            buckets[max_cov].append(ki)
            
        current_max = max_cov
        blocks = []
        
        while uncovered_count > 0:
            # Find the highest non-empty bucket
            while current_max >= 0 and not buckets[current_max]:
                current_max -= 1
            
            if current_max < 0:
                break
                
            ki = buckets[current_max].pop()
            
            if selected[ki]:
                continue
                
            if coverage[ki] != current_max:
                # Stale entry, move to correct bucket
                buckets[coverage[ki]].append(ki)
                continue
                
            if coverage[ki] == 0:
                continue
            
            # Select this k-subset
            selected[ki] = True
            blocks.append(list(all_k_subsets[ki]))
            
            # Update coverage for neighbors
            for ti in k_to_t_indices[ki]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1
                    for other_ki in t_to_k_indices[ti]:
                        if not selected[other_ki]:
                            coverage[other_ki] -= 1
                            
        return blocks

    # Run multiple times with different orders
    best_blocks = None
    best_len = float('inf')
    
    # Deterministic run
    order = list(range(num_k))
    blocks = run_greedy(order)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = blocks
        
    # Randomized runs
    num_restarts = 30
    for _ in range(num_restarts):
        random_order = list(range(num_k))
        random.shuffle(random_order)
        blocks = run_greedy(random_order)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
