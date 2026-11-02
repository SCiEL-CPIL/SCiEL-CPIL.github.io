import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    
    block_subsets = []
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, block in enumerate(all_k_blocks):
        covered_ids = []
        for t_sub in itertools.combinations(block, t):
            sid = t_subset_to_id[t_sub]
            covered_ids.append(sid)
            subset_to_blocks[sid].append(i)
        block_subsets.append(covered_ids)
        
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline - 10.0:
        uncovered_set = set(range(num_t_subsets))
        selected_blocks = []
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        
        while uncovered_set:
            if time.time() > deadline - 5.0:
                break
                
            # Pick random uncovered subset
            sid = random.choice(list(uncovered_set))
            
            candidates = subset_to_blocks[sid]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for s in block_subsets[b_idx]:
                    if s in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            
            for s in block_subsets[best_block_idx]:
                counts[s] += 1
                uncovered_set.discard(s)
                
        # Local Search
        in_use = set(selected_blocks)
        
        def is_removable(block_idx):
            for sid in block_subsets[block_idx]:
                if counts[sid] == 1:
                    return False
            return True
            
        def find_replacement(block_idx):
            critical = [sid for sid in block_subsets[block_idx] if counts[sid] == 1]
            if not critical:
                return -1
                
            critical.sort(key=lambda s: len(subset_to_blocks[s]))
            
            candidates = set(subset_to_blocks[critical[0]])
            candidates.discard(block_idx)
            
            for sid in critical[1:]:
                candidates &= set(subset_to_blocks[sid])
                if not candidates:
                    return -1
                    
            if not candidates:
                return -1
                
            best_cand = -1
            best_score = -1
            for cand in candidates:
                score = np.sum(counts[block_subsets[cand]])
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    
            return best_cand

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            for b in list(in_use):
                if time.time() > deadline - 2.0:
                    break
                if is_removable(b):
                    for sid in block_subsets[b]:
                        counts[sid] -= 1
                    in_use.remove(b)
                    improved = True
                    
            for b in list(in_use):
                if time.time() > deadline - 2.0:
                    break
                rep = find_replacement(b)
                if rep != -1:
                    for sid in block_subsets[b]:
                        counts[sid] -= 1
                    for sid in block_subsets[rep]:
                        counts[sid] += 1
                    in_use.remove(b)
                    in_use.add(rep)
                    improved = True
                    
        if len(in_use) < best_size:
            best_size = len(in_use)
            best_blocks = in_use.copy()
            
    # Final Output
    result = []
    for b_idx in best_blocks:
        result.append(list(all_k_blocks[b_idx]))
        
    return result
