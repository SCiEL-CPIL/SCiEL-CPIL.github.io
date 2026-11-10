import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Randomized Greedy 
    followed by Iterated Local Search (Perturbation + Deletion).
    """
    points = list(range(v))
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    num_blocks = len(all_k_subsets)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each block, the indices of t-subsets it covers
    block_t_indices = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, bs in enumerate(all_k_subsets):
        t_indices = [t_to_idx[ts] for ts in itertools.combinations(bs, t)]
        block_t_indices[i] = t_indices
        for ti in t_indices:
            t_to_blocks[ti].append(i)
            
    # Helper to perform greedy deletion
    def greedy_delete(block_indices):
        if not block_indices:
            return set()
            
        # Initial counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in block_indices:
            for ti in block_t_indices[bi]:
                counts[ti] += 1
                
        current_blocks = set(block_indices)
        
        # Repeatedly remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Find a redundant block
            for bi in list(current_blocks):
                redundant = True
                for ti in block_t_indices[bi]:
                    if counts[ti] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    current_blocks.remove(bi)
                    for ti in block_t_indices[bi]:
                        counts[ti] -= 1
                    changed = True
                    break # Restart scan
        
        return current_blocks

    # Helper to generate a randomized greedy cover
    def randomized_greedy_cover_fast():
        uncovered = set(range(num_t_subsets))
        current_blocks = set()
        
        while uncovered:
            # Pick a random uncovered t-subset
            target_t = random.choice(list(uncovered))
            # Pick a random block covering it
            candidate_blocks = t_to_blocks[target_t]
            chosen_block = random.choice(candidate_blocks)
            
            current_blocks.add(chosen_block)
            
            # Update uncovered
            for ti in block_t_indices[chosen_block]:
                if ti in uncovered:
                    uncovered.discard(ti)
                    
        return current_blocks

    start_time = time.time()
    time_limit = 100
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple iterations of Randomized Greedy + Deletion
    while time.time() - start_time < time_limit:
        # 1. Generate new random cover
        rand_blocks = randomized_greedy_cover_fast()
        reduced_blocks = greedy_delete(rand_blocks)
        
        if len(reduced_blocks) < best_size:
            best_size = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        # 2. Perturbation: Take best, add random blocks, delete
        if best_blocks and time.time() - start_time < time_limit:
            perturbed = set(best_blocks)
            # Add 5-15 random blocks to escape local minima
            num_add = random.randint(5, 15)
            for _ in range(num_add):
                rb = random.randint(0, num_blocks - 1)
                perturbed.add(rb)
            
            reduced_perturbed = greedy_delete(perturbed)
            if len(reduced_perturbed) < best_size:
                best_size = len(reduced_perturbed)
                best_blocks = reduced_perturbed

    # Convert indices back to lists of points
    if best_blocks:
        return [list(all_k_subsets[i]) for i in best_blocks]
    else:
        return []
