import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # block_triple_indices[i] is a list of triple indices in block i
    # triple_to_blocks[j] is a list of block indices containing triple j
    block_triple_indices = [None] * n_blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        idxs = [triple_to_idx[sub] for sub in subs]
        block_triple_indices[i] = idxs
        for idx in idxs:
            triple_to_blocks[idx].append(i)

    def greedy_cover(random_seed, deadline):
        rng = np.random.RandomState(random_seed)
        # gain[i] is the number of uncovered triples in block i
        gain = np.full(n_blocks, k_t, dtype=np.int32)
        covered = np.zeros(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline:
                break
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly pick one
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected_blocks.append(best_idx)
            
            # Update gains for blocks sharing triples with best_idx
            for triple_idx in block_triple_indices[best_idx]:
                if not covered[triple_idx]:
                    covered[triple_idx] = True
                    remaining -= 1
                    # Decrement gain for all blocks containing this triple
                    for b_idx in triple_to_blocks[triple_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks

    def local_search_remove(block_indices, deadline):
        """Try to remove blocks that are redundant (all their triples are covered by other blocks)."""
        if not block_indices:
            return block_indices
        
        # Build coverage: for each triple, which selected blocks cover it
        n_sel = len(block_indices)
        # triple_covered_by: for each triple, count of selected blocks covering it
        triple_covered_by = np.zeros(n_triples, dtype=np.int32)
        block_triples_in_sel = []
        
        for bi in block_indices:
            idxs = block_triple_indices[bi]
            block_triples_in_sel.append(idxs)
            for idx in idxs:
                triple_covered_by[idx] += 1
        
        # Try removing blocks in random order
        rng = np.random.RandomState(42)
        order = list(range(n_sel))
        rng.shuffle(order)
        
        kept = [True] * n_sel
        current_count = n_sel
        
        for pos in order:
            if time.time() > deadline:
                break
            if not kept[pos]:
                continue
            
            # Check if all triples in this block are covered by at least one other kept block
            can_remove = True
            for idx in block_triples_in_sel[pos]:
                if triple_covered_by[idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                kept[pos] = False
                current_count -= 1
                for idx in block_triples_in_sel[pos]:
                    triple_covered_by[idx] -= 1
        
        return [block_indices[i] for i in range(n_sel) if kept[i]]

    # Run multiple times
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Since the algorithm is faster, we can do more restarts
    # 50 restarts in the old code might have taken a long time.
    # Let's try 50 restarts to see if we can find a better solution.
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        blocks_idx = greedy_cover(i, deadline)
        
        # Apply local search to try to remove redundant blocks
        if time.time() < deadline:
            blocks_idx = local_search_remove(blocks_idx, deadline)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = [list(all_blocks[bi]) for bi in blocks_idx]
            
    # If we haven't found any valid solution yet (shouldn't happen), fall back
    if best_blocks is None:
        blocks_idx = greedy_cover(0, deadline)
        best_blocks = [list(all_blocks[bi]) for bi in blocks_idx]
        
    return best_blocks
