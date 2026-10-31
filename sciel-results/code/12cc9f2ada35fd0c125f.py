import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # Precomputation
    points = range(v)
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(k_blocks)
    
    # Precompute coverage: for each block, which t-subsets does it cover?
    block_cover = [None] * num_k_blocks
    for b_idx, block in enumerate(k_blocks):
        covered = []
        for sub in itertools.combinations(block, t):
            covered.append(subset_to_idx[sub])
        block_cover[b_idx] = covered
        
    # Precompute for each t-subset, which blocks cover it?
    subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx in range(num_k_blocks):
        for s_idx in block_cover[b_idx]:
            subset_blocks[s_idx].append(b_idx)
            
    best_cover = None
    best_len = float('inf')
    
    def run_greedy_reduction(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        count = [0] * num_t_subsets
        uncovered = set(range(num_t_subsets))
        cover = []
        
        # Greedy Phase
        while uncovered:
            # Pick a random uncovered subset
            # Converting set to list for random.choice is O(N)
            # N is ~3876, so it's fast.
            s_idx = random.choice(list(uncovered))
            
            best_b_idx = -1
            best_gain = -1
            
            # Candidates are blocks covering s_idx
            for b_idx in subset_blocks[s_idx]:
                gain = 0
                for sub_idx in block_cover[b_idx]:
                    if count[sub_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            if best_b_idx == -1:
                break
                
            cover.append(best_b_idx)
            for sub_idx in block_cover[best_b_idx]:
                count[sub_idx] += 1
                if count[sub_idx] == 1:
                    uncovered.discard(sub_idx)
                    
        # Reduction Phase
        changed = True
        while changed:
            changed = False
            for i in range(len(cover) - 1, -1, -1):
                b_idx = cover[i]
                is_redundant = True
                for sub_idx in block_cover[b_idx]:
                    if count[sub_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    cover.pop(i)
                    for sub_idx in block_cover[b_idx]:
                        count[sub_idx] -= 1
                    changed = True
                    break
            
        return cover

    # Run multiple iterations
    # First run with a fixed seed to ensure we have a baseline
    cover = run_greedy_reduction(42)
    if cover is not None:
        best_cover = cover
        best_len = len(cover)
        
    iter_count = 1
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        
        seed = iter_count
        cover = run_greedy_reduction(seed)
        if cover is not None:
            curr_len = len(cover)
            if curr_len < best_len:
                best_len = curr_len
                best_cover = cover
        
        iter_count += 1
        
    if best_cover is None:
        return []
        
    result = [list(k_blocks[i]) for i in best_cover]
    return result
