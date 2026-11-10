# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)
    
    # 1. Enumerate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage masks for all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # masks[i] is a bitmask where bit j is 1 if t-subset j is covered by k-subset i
    masks = [0] * n_k
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
        masks[i] = m

    # 3. Greedy Set Cover with Local Search
    uncovered = (1 << n_t) - 1
    best_solution = None
    best_len = float('inf')
    
    # We will run a few randomized greedy attempts
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(attempt * 7919 + 13)
        
        current_uncovered = uncovered
        selected = []
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            
            # Scan all k-subsets
            for i in range(n_k):
                gain_mask = current_uncovered & masks[i]
                gain = gain_mask.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    # Random tie-break
                    if rng.random() < 0.2:
                        best_gain = gain
                        best_idx = i
                        best_mask = gain_mask
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 10:
                break
        
        if current_uncovered == 0:
            blocks = [list(all_k_subsets[i]) for i in selected]
            
            if len(selected) < best_len:
                best_len = len(selected)
                best_solution = blocks
            
            # Local Search: Try to remove redundant blocks
            if time.time() < deadline - 5:
                block_t_indices = []
                for b in blocks:
                    indices = []
                    for ts in itertools.combinations(b, t):
                        indices.append(t_to_idx[ts])
                    block_t_indices.append(indices)
                
                cov_counts = [0] * n_t
                for indices in block_t_indices:
                    for idx in indices:
                        cov_counts[idx] += 1
                
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    order = list(range(len(blocks)))
                    rng.shuffle(order)
                    
                    for bi in order:
                        if bi >= len(blocks):
                            continue
                        
                        redundant = True
                        for idx in block_t_indices[bi]:
                            if cov_counts[idx] == 1:
                                redundant = False
                                break
                        
                        if redundant:
                            for idx in block_t_indices[bi]:
                                cov_counts[idx] -= 1
                            
                            blocks.pop(bi)
                            block_t_indices.pop(bi)
                            improved = True
                            break
                
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_solution = blocks

    if best_solution is None:
        # Fallback: Greedy without local search
        current_uncovered = uncovered
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            for i in range(n_k):
                gain = (current_uncovered & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            current_uncovered &= ~masks[best_idx]
        best_solution = [list(all_k_subsets[i]) for i in selected]
        
    return best_solution
# EVOLVE-BLOCK-END
