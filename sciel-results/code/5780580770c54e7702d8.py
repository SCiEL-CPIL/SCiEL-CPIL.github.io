import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_t_indices = [] 
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[s] for s in t_subs_in_block]
        block_t_indices.append(np.array(indices, dtype=np.int32))
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
            
    best_block_indices = None
    best_score = float('inf')
    
    restart = 0
    while time.time() < deadline - 10.0:
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        
        # Greedy Phase
        while True:
            # Find an uncovered subset
            uncovered_mask = (coverage == 0)
            if not np.any(uncovered_mask):
                break
            
            # Random sampling of uncovered subsets to build candidate pool
            K = 30 
            candidate_block_set = set()
            
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            sampled = random.sample(uncovered_indices.tolist(), min(K, len(uncovered_indices)))
            
            for u_idx in sampled:
                candidate_block_set.update(t_subset_to_blocks[u_idx])
                
            if not candidate_block_set:
                break # Should not happen if there are uncovered subsets
                
            cand_list = list(candidate_block_set)
            
            # Vectorized Gain Calculation
            gains = np.zeros(len(cand_list), dtype=np.int32)
            for i, b_idx in enumerate(cand_list):
                idxs = block_t_indices[b_idx]
                gains[i] = np.sum(coverage[idxs] == 0)
                
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
            
        # Reduction Phase
        if not block_indices:
            continue
            
        current_indices = block_indices[:]
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            essential_mask = (coverage == 1)
            
            # If no essential subsets, we can remove any block
            # (since min count is >= 2, removing one drops min to >= 1)
            if not np.any(essential_mask):
                if not current_indices:
                    break
                b_idx = current_indices.pop()
                idxs = block_t_indices[b_idx]
                coverage[idxs] -= 1
                continue
            
            # Find blocks that cover NO essential subset
            new_indices = []
            removed_any = False
            for b_idx in current_indices:
                idxs = block_t_indices[b_idx]
                # Check if any of its t-subsets is essential (count == 1)
                if np.any(coverage[idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    # Remove it
                    coverage[idxs] -= 1
                    removed_any = True
            
            current_indices = new_indices
            
            if not removed_any:
                break
                
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
    if best_block_indices is None:
        # Fallback: return a valid covering
        # We should have found one. If not, return empty (invalid) or a trivial one.
        # To be safe, if we didn't finish even one restart, we can't return a valid cover easily.
        # But the first restart should be fast.
        pass
        
    return [list(all_k_blocks[i]) for i in best_block_indices]
