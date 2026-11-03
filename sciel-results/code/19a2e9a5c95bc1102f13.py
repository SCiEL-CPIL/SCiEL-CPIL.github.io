import itertools
import numpy as np
import time
import math

_PRECOMPUTED = None

def _precompute(v, k, t):
    global _PRECOMPUTED
    if _PRECOMPUTED and _PRECOMPUTED['v'] == v and _PRECOMPUTED['k'] == k and _PRECOMPUTED['t'] == t:
        return _PRECOMPUTED
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tup: i for i, tup in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    num_triples_per_block = math.comb(k, t)
    
    block_triples = np.zeros((num_blocks, num_triples_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        block_triples[i] = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
            
    triple_to_blocks_sets = [set(lst) for lst in triple_to_blocks]
    
    _PRECOMPUTED = {
        'v': v, 'k': k, 't': t,
        'all_blocks': all_blocks,
        'block_triples': block_triples,
        'triple_to_blocks_sets': triple_to_blocks_sets,
        'num_blocks': num_blocks,
        'num_triples': num_triples
    }
    return _PRECOMPUTED

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0
    
    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    block_triples = data['block_triples']
    triple_to_blocks_sets = data['triple_to_blocks_sets']
    num_triples = data['num_triples']
    
    triple_counts = np.zeros(num_triples, dtype=np.int32)
    current_block_indices = []
    
    while np.any(triple_counts == 0):
        gains = np.sum(triple_counts[block_triples] == 0, axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        best_block_idx = candidates[0]
        current_block_indices.append(best_block_idx)
        triple_counts[block_triples[best_block_idx]] += 1
        
        if time.time() > deadline:
            break
            
    current_set = set(current_block_indices)
    
    while time.time() < deadline:
        improved = False
        
        for i in range(len(current_block_indices)):
            if time.time() > deadline:
                break
                
            b_idx = current_block_indices[i]
            b_triples = block_triples[b_idx]
            
            unique_triples = b_triples[triple_counts[b_triples] == 1]
            
            if len(unique_triples) == 0:
                del current_block_indices[i]
                current_set.remove(b_idx)
                triple_counts[b_triples] -= 1
                improved = True
                break
                
            if len(unique_triples) > 0:
                candidates_set = triple_to_blocks_sets[unique_triples[0]]
                for t_idx in unique_triples[1:]:
                    candidates_set = candidates_set.intersection(triple_to_blocks_sets[t_idx])
                    
                candidates_set.difference_update(current_set)
                
                if candidates_set:
                    best_candidate = -1
                    best_gain = 0
                    
                    for cand_idx in candidates_set:
                        cand_triples = block_triples[cand_idx]
                        gain = np.sum(triple_counts[cand_triples] == 0)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = cand_idx
                            
                    if best_candidate != -1 and best_gain > 0:
                        triple_counts[b_triples] -= 1
                        current_set.remove(b_idx)
                        
                        triple_counts[block_triples[best_candidate]] += 1
                        current_set.add(best_candidate)
                        
                        current_block_indices[i] = best_candidate
                        improved = True
                        break 

        if not improved:
            break
            
    return [list(all_blocks[i]) for i in current_block_indices]
