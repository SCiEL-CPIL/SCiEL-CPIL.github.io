import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # Precompute block to triples (as sets for fast membership) and lists for numpy
    block_to_triples_set = []
    block_to_triples_list = []
    for block in all_k_subsets:
        triples_list = [t_sub_index[ts] for ts in itertools.combinations(block, t)]
        block_to_triples_list.append(triples_list)
        block_to_triples_set.append(frozenset(triples_list))
        
    # Map triple to list of blocks
    triple_to_blocks = defaultdict(list)
    for b_idx, triples in enumerate(block_to_triples_set):
        for t_idx in triples:
            triple_to_blocks[t_idx].append(b_idx)
            
    # Greedy initialization using numpy for speed
    cover_matrix = np.array(block_to_triples_list, dtype=np.int32)
    
    def get_greedy_initial(seed):
        random.seed(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        while np.any(uncovered):
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0: break
            candidates = np.where(gains == max_gain)[0]
            chosen = candidates[random.randint(0, len(candidates)-1)]
            blocks.append(chosen)
            uncovered[cover_matrix[chosen]] = False
        return blocks

    def optimize(blocks):
        block_set = set(blocks)
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in block_set:
            counts[block_to_triples_list[b_idx]] += 1
            
        improved = True
        while improved:
            improved = False
            # Remove redundant
            to_remove = []
            for b_idx in block_set:
                if np.min(counts[block_to_triples_list[b_idx]]) > 1:
                    to_remove.append(b_idx)
            for b_idx in to_remove:
                block_set.remove(b_idx)
                counts[block_to_triples_list[b_idx]] -= 1
                improved = True
            
            # Swap
            for b_idx in list(block_set):
                critical = [t_idx for t_idx in block_to_triples_set[b_idx] if counts[t_idx] == 1]
                if not critical:
                    continue
                critical.sort(key=lambda x: len(triple_to_blocks[x]))
                first_crit = critical[0]
                candidates = triple_to_blocks[first_crit]
                
                found = False
                for b_j in candidates:
                    if b_j in block_set:
                        continue
                    # Check if b_j covers all critical
                    valid = True
                    for t_idx in critical:
                        if t_idx not in block_to_triples_set[b_j]:
                            valid = False
                            break
                    if valid:
                        block_set.remove(b_idx)
                        counts[block_to_triples_list[b_idx]] -= 1
                        block_set.add(b_j)
                        counts[block_to_triples_list[b_j]] += 1
                        improved = True
                        found = True
                        break
                if found:
                    break
            
        return list(block_set)

    best_blocks = []
    best_count = float('inf')
    start_time = time.time()
    timeout = 115
    
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
        
        init_blocks = get_greedy_initial(restart)
        opt_blocks = optimize(init_blocks)
        
        if len(opt_blocks) < best_count:
            best_count = len(opt_blocks)
            best_blocks = opt_blocks
            
    return [list(all_k_subsets[i]) for i in best_blocks]
