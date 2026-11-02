import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    # Lookup table for triples
    # lookup[a, b, c] = index of (a,b,c) in all_t_subsets
    lookup = np.full((v, v, v), -1, dtype=np.int32)
    for idx, (a, b, c) in enumerate(all_t_subsets):
        lookup[a, b, c] = idx

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Convert to numpy array
    points_matrix = np.array(all_k_subsets, dtype=np.int32)
    
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    sub_indices = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    # Vectorized fill of sub_indices
    pos_combinations = list(itertools.combinations(range(k), t))
    for j, pos in enumerate(pos_combinations):
        # Extract columns
        cols = points_matrix[:, list(pos)]
        # cols shape: (n_k_subsets, t)
        # We need to index lookup with these columns
        # lookup[cols[:, 0], cols[:, 1], cols[:, 2]]
        if t == 3:
            sub_indices[:, j] = lookup[cols[:, 0], cols[:, 1], cols[:, 2]]
        elif t == 2:
             sub_indices[:, j] = lookup[cols[:, 0], cols[:, 1], 0] # Need 2D lookup or adjust
        # Assuming t=3 for this task, but let's make it generic if possible.
        # For t=3, the above works.
        
    # For t != 3, we'd need a different lookup structure. 
    # Given the task is C(24,7,3), t=3 is fixed. 
    # I will assume t=3 for optimization, but keep a fallback if t is different.
    if t != 3:
        # Fallback to slow method if t != 3
        sub_indices = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
        for i, block in enumerate(all_k_subsets):
            for j, triple in enumerate(itertools.combinations(block, t)):
                sub_indices[i, j] = triple_to_idx[triple]

    # Map from block tuple to index
    block_to_idx = {block: i for i, block in enumerate(all_k_subsets)}

    def fast_greedy_cover(seed, randomize_ties=True):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        selected_indices = []
        used_mask = np.zeros(n_k_subsets, dtype=bool)
        
        while uncovered.any():
            if time.time() > deadline - 5:
                break
                
            gains = np.sum(uncovered[sub_indices], axis=1)
            gains[used_mask] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if randomize_ties and len(candidates) > 1:
                chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                chosen_idx = candidates[0]
                
            selected_indices.append(chosen_idx)
            used_mask[chosen_idx] = True
            uncovered[sub_indices[chosen_idx]] = False
            
        return selected_indices

    def reduce_cover(block_indices):
        if not block_indices:
            return []
            
        coverage = np.zeros(n_triples, dtype=np.int32)
        for idx in block_indices:
            coverage[sub_indices[idx]] += 1
            
        current_blocks = list(block_indices)
        improved = True
        
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            new_blocks = []
            for idx in current_blocks:
                subs = sub_indices[idx]
                if np.min(coverage[subs]) > 1:
                    coverage[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(idx)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def local_search(block_indices, seed):
        rng = random.Random(seed)
        current_blocks = list(block_indices)
        current_blocks = reduce_cover(current_blocks)
        
        def get_coverage(bks):
            cov = np.zeros(n_triples, dtype=np.int32)
            for idx in bks:
                cov[sub_indices[idx]] += 1
            return cov
        
        coverage = get_coverage(current_blocks)
        used_mask = np.zeros(n_k_subsets, dtype=bool)
        for idx in current_blocks:
            used_mask[idx] = True
            
        max_kick_iterations = 50
        for _ in range(max_kick_iterations):
            if time.time() > deadline - 5:
                break
                
            if not current_blocks:
                break
                
            kick_idx_pos = rng.randint(0, len(current_blocks) - 1)
            kick_idx = current_blocks[kick_idx_pos]
            
            kick_subs = sub_indices[kick_idx]
            
            # Check if redundant
            if np.min(coverage[kick_subs]) > 1:
                coverage[kick_subs] -= 1
                current_blocks.pop(kick_idx_pos)
                used_mask[kick_idx] = False
                continue
            
            # Kick
            coverage[kick_subs] -= 1
            uncovered = (coverage == 0)
            
            if not np.any(uncovered):
                current_blocks.pop(kick_idx_pos)
                used_mask[kick_idx] = False
                continue
            
            gains = np.sum(uncovered[sub_indices], axis=1)
            gains[used_mask] = -1
            gains[kick_idx] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                coverage[kick_subs] += 1
                continue
                
            candidates = np.where(gains == max_gain)[0]
            new_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            new_subs = sub_indices[new_idx]
            coverage[new_subs] += 1
            
            if np.any(coverage == 0):
                coverage[new_subs] -= 1
                continue
            else:
                current_blocks[kick_idx_pos] = new_idx
                used_mask[kick_idx] = False
                used_mask[new_idx] = True
                
                current_blocks = reduce_cover(current_blocks)
                coverage = get_coverage(current_blocks)
                used_mask = np.zeros(n_k_subsets, dtype=bool)
                for idx in current_blocks:
                    used_mask[idx] = True
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    restart_count = 0
    
    while time.time() < deadline - 5:
        blocks = fast_greedy_cover(restart_count, randomize_ties=True)
        blocks = local_search(blocks, seed=restart_count)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
        restart_count += 1
        
    if best_blocks is None:
        best_blocks = fast_greedy_cover(0, randomize_ties=False)
        
    return [list(all_k_subsets[i]) for i in best_blocks]
