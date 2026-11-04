import itertools
import numpy as np
from math import comb
from collections import defaultdict

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # Precompute blocks covering each triple for local search
    # blocks_covering_triple[i] is a list of block indices covering triple i
    blocks_covering_triple = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for j in range(k_t):
            t_idx = block_triple_indices[i, j]
            blocks_covering_triple[t_idx].append(i)

    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            block_idx = block_triple_indices[best_idx]
            new_uncovered = uncovered_mask[block_idx]
            remaining -= np.sum(new_uncovered)
            uncovered_mask[block_idx] = False
            selected_blocks.append(best_idx)
            
        return selected_blocks

    def local_search_2_for_1(block_indices):
        # block_indices is a list of indices into all_blocks
        if len(block_indices) < 2:
            return block_indices
            
        current_blocks = set(block_indices)
        n_current = len(current_blocks)
        
        # Calculate coverage counts
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_triple_indices[b_idx]] += 1
            
        improved = True
        while improved:
            improved = False
            # Iterate through all pairs in current_blocks
            # To avoid modifying while iterating, we can collect swaps or break on first swap
            # We want to find ANY swap.
            block_list = list(current_blocks)
            for i in range(len(block_list)):
                for j in range(i + 1, len(block_list)):
                    b1 = block_list[i]
                    b2 = block_list[j]
                    
                    # Find unique triples covered only by b1 or b2
                    # A triple t is unique if cover_count[t] == 1 and t is in b1 or b2
                    # Since we only remove b1 and b2, any triple with count 1 in the current set
                    # that is in b1 or b2 will become uncovered.
                    
                    # Get triples in b1
                    t1 = block_triple_indices[b1]
                    t2 = block_triple_indices[b2]
                    
                    # Combine and check counts
                    # We need to find if there is a block b_new covering all such triples.
                    # Let U be the set of triples with count 1 in b1 ∪ b2.
                    
                    # Optimization: 
                    # If there are no triples with count 1 in b1, and none in b2, 
                    # then removing b1 and b2 doesn't uncover anything? 
                    # No, if a triple is in both b1 and b2, its count is at least 2 (from b1 and b2).
                    # So if count is 1, it must be in exactly one of them.
                    
                    unique_triples = []
                    for t_idx in t1:
                        if cover_count[t_idx] == 1:
                            unique_triples.append(t_idx)
                    for t_idx in t2:
                        if cover_count[t_idx] == 1:
                            unique_triples.append(t_idx)
                            
                    if not unique_triples:
                        # We can remove both? No, we want to replace with 1.
                        # If unique_triples is empty, it means all triples in b1 and b2 are covered by others.
                        # So we can just remove b1 and b2? 
                        # That would reduce count by 2.
                        # But wait, if unique_triples is empty, b1 and b2 are redundant.
                        # Standard greedy produces minimal covers, so this shouldn't happen?
                        # Unless the greedy was not minimal? 
                        # My proof said greedy is minimal. 
                        # So unique_triples should not be empty if we remove 2 blocks?
                        # Actually, if b1 and b2 share no "unique" responsibility, we can remove them.
                        # But we want to replace 2 with 1. 
                        # If unique_triples is empty, we can remove 2 and add 0. 
                        # That's better than 2->1.
                        # However, the problem is set cover, not exact cover.
                        # If we can remove 2, we should.
                        # Let's handle the case where we can remove 1 or 2 blocks.
                        # But 2-for-1 is the target.
                        # If unique_triples is empty, we can just remove b1 and b2.
                        # Let's add this logic.
                        
                        # Remove b1 and b2
                        current_blocks.remove(b1)
                        current_blocks.remove(b2)
                        for t_idx in t1:
                            cover_count[t_idx] -= 1
                        for t_idx in t2:
                            cover_count[t_idx] -= 1
                        improved = True
                        break
                            
                    # If unique_triples is not empty, we need a block covering all of them.
                    # Find intersection of blocks covering each unique triple.
                    candidate_blocks = set(blocks_covering_triple[unique_triples[0]])
                    for t_idx in unique_triples[1:]:
                        candidate_blocks.intersection_update(blocks_covering_triple[t_idx])
                        
                    # Remove blocks already in current_blocks
                    candidate_blocks.difference_update(current_blocks)
                    
                    if candidate_blocks:
                        # Pick one candidate
                        b_new = next(iter(candidate_blocks))
                        
                        # Perform swap
                        current_blocks.remove(b1)
                        current_blocks.remove(b2)
                        current_blocks.add(b_new)
                        
                        # Update counts
                        for t_idx in t1:
                            cover_count[t_idx] -= 1
                        for t_idx in t2:
                            cover_count[t_idx] -= 1
                        for t_idx in block_triple_indices[b_new]:
                            cover_count[t_idx] += 1
                            
                        improved = True
                        break
                if improved:
                    break
            # Restart the pair iteration if improved
            
        return list(current_blocks)

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 20
    
    for i in range(n_restarts):
        blocks_idx = greedy_cover(i)
        # Apply local search
        optimized_idx = local_search_2_for_1(blocks_idx)
        
        if len(optimized_idx) < best_len:
            best_len = len(optimized_idx)
            best_blocks = [list(all_blocks[idx]) for idx in optimized_idx]
            
    return best_blocks
