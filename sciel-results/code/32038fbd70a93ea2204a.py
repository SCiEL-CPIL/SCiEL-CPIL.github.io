import itertools
import numpy as np
import time

def construct(v, k, t):
    # Pre-computation specific to the problem structure
    # Map triples to indices for fast lookup
    triple_to_idx = {}
    all_triples = list(itertools.combinations(range(v), t))
    for i, tr in enumerate(all_triples):
        triple_to_idx[tr] = i
        
    # Generate all blocks and their corresponding triple indices
    blocks_points = []
    blocks_triples_list = []
    
    for block in itertools.combinations(range(v), k):
        blocks_points.append(list(block))
        triples_in_block = []
        for tr in itertools.combinations(block, t):
            triples_in_block.append(triple_to_idx[tr])
        blocks_triples_list.append(triples_in_block)
        
    N_blocks = len(blocks_points)
    if N_blocks == 0:
        return []
        
    blocks_triples = np.array(blocks_triples_list, dtype=np.int32)
    n_triples = len(all_triples)
    
    best_solution = None
    start_time = time.time()
    
    while time.time() - start_time < 140:
        cover_count = np.zeros(n_triples, dtype=np.int32)
        chosen_blocks = []
        is_available = np.ones(N_blocks, dtype=bool)
        
        # Greedy construction
        while np.sum(cover_count == 0) > 0:
            is_uncovered = (cover_count == 0)
            # Calculate scores: number of uncovered triples in each available block
            scores = is_uncovered[blocks_triples].sum(axis=1)
            scores[~is_available] = -1
            
            max_score = np.max(scores)
            if max_score == -1:
                break # No more blocks available (should not happen)
            
            # Pick randomly from top candidates to allow restarts to differ
            # If max_score is high, ties are likely.
            candidates = np.where(scores == max_score)[0]
            idx = candidates[np.random.randint(len(candidates))]
            
            chosen_blocks.append(idx)
            cover_count[blocks_triples[idx]] += 1
            is_available[idx] = False
            
        # Redundancy removal
        # Rebuild cover count for chosen blocks to check redundancy
        cover_count_check = np.zeros(n_triples, dtype=np.int32)
        for b in chosen_blocks:
            cover_count_check[blocks_triples[b]] += 1
            
        kept_blocks = []
        for b in chosen_blocks:
            # Tentatively remove block b
            cover_count_check[blocks_triples[b]] -= 1
            # If any triple in b now has count 0, b is critical
            if np.any(cover_count_check[blocks_triples[b]] == 0):
                kept_blocks.append(b)
            # Else, b is redundant, discard it
        
        if len(kept_blocks) == 0:
            break
            
        if best_solution is None or len(kept_blocks) < len(best_solution):
            best_solution = kept[:]
            # If we find a solution with size <= 69, it's optimal (LB).
            # If we find < 76, it's better than known.
            # We continue searching for better.
            
    if best_solution is None:
        # Fallback: return empty or a trivial construction if loop didn't finish
        # This should not happen
        return []
        
    return [blocks_points[i] for i in best_solution]
