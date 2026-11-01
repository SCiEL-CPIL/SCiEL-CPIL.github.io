import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    # Precompute all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(all_points, k))
    num_blocks = len(all_blocks)
    
    # For each block, store the indices of the t-subsets it covers
    block_triples = []
    # For each triple, store the indices of the blocks that cover it
    triple_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        triples_in_block = []
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[tuple(triple)]
            triples_in_block.append(t_idx)
            triple_blocks[t_idx].append(b_idx)
        block_triples.append(triples_in_block)
        
    best_blocks = None
    best_len = float('inf')
    
    def random_greedy_cover():
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks_in_sol = []
        blocks_in_sol_set = set()
        
        while True:
            if not np.any(uncovered):
                break
            
            # Pick a random uncovered triple
            seed_idx = None
            while True:
                candidate = random.randint(0, num_triples - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            
            candidates = triple_blocks[seed_idx]
            
            if not candidates:
                break # Should not happen if v,k,t are valid
                
            best_candidate = -1
            best_gain = -1
            
            # Evaluate candidates
            for b_idx in candidates:
                if b_idx in blocks_in_sol_set:
                    continue
                gain = 0
                for t_idx in block_triples[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
                    if gain == num_triples: # Perfect block
                        break
            
            if best_candidate == -1:
                # Fallback: pick first candidate
                best_candidate = candidates[0]
                
            blocks_in_sol.append(best_candidate)
            blocks_in_sol_set.add(best_candidate)
            
            for t_idx in block_triples[best_candidate]:
                uncovered[t_idx] = 0
                
        return blocks_in_sol

    def local_search(blocks_in_sol):
        if not blocks_in_sol:
            return blocks_in_sol
            
        # counts[t_idx] = number of blocks in sol covering t_idx
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks_in_sol:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
                
        sol_set = set(blocks_in_sol)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            to_remove = []
            for b_idx in blocks_in_sol:
                is_redundant = True
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(b_idx)
                    
            if to_remove:
                remove_set = set(to_remove)
                blocks_in_sol = [b for b in blocks_in_sol if b not in remove_set]
                sol_set = set(blocks_in_sol)
                for b_idx in to_remove:
                    for t_idx in block_triples[b_idx]:
                        counts[t_idx] -= 1
                improved = True
                continue # Restart loop after removals
            
            # 2. Try swaps to enable removals
            # For each block, check if it has unique triples
            # If so, look for a replacement that covers those unique triples
            # and see if it makes another block redundant
            
            swaps_made = False
            # Shuffle to randomize order
            for b_idx in random.sample(blocks_in_sol, len(blocks_in_sol)):
                unique_triples = []
                for t_idx in block_triples[b_idx]:
                    if counts[t_idx] == 1:
                        unique_triples.append(t_idx)
                        
                if not unique_triples:
                    continue # Redundant, should have been removed
                
                # Find a block B not in sol that covers all unique_triples
                # We can iterate through blocks that cover the first unique triple
                candidate_replacements = []
                if unique_triples:
                    first_t = unique_triples[0]
                    for cand_b_idx in triple_blocks[first_t]:
                        if cand_b_idx in sol_set:
                            continue
                        # Check if cand covers all unique_triples
                        valid = True
                        for ut in unique_triples:
                            if ut not in block_triples[cand_b_idx]:
                                valid = False
                                break
                        if valid:
                            candidate_replacements.append(cand_b_idx)
                
                if not candidate_replacements:
                    continue
                
                # Try each replacement
                # To save time, maybe just try a few random ones
                random.shuffle(candidate_replacements)
                for new_b_idx in candidate_replacements[:5]: # Limit checks
                    # Tentative swap: remove b_idx, add new_b_idx
                    # Check if any block becomes redundant
                    # Update counts virtually
                    # counts_new = counts - block_triples[b_idx] + block_triples[new_b_idx]
                    
                    # Optimization: Only check blocks that might become redundant
                    # A block C becomes redundant if for all t in C, counts_new[t] > 1
                    # This is expensive to check fully.
                    # Heuristic: Check if new_b_idx covers any triple that was uniquely covered by b_idx?
                    # We already know it covers unique_triples of b_idx.
                    
                    # Let's just do the swap and check redundancy in the main loop
                    # But we need to know if it's "better".
                    # If we swap, and then a removal happens, we win.
                    
                    # Perform swap
                    sol_set.remove(b_idx)
                    sol_set.add(new_b_idx)
                    blocks_in_sol.remove(b_idx)
                    blocks_in_sol.append(new_b_idx)
                    
                    for t_idx in block_triples[b_idx]:
                        counts[t_idx] -= 1
                    for t_idx in block_triples[new_b_idx]:
                        counts[t_idx] += 1
                        
                    # Check if any block is now redundant
                    found_redundant = False
                    for c_idx in blocks_in_sol:
                        is_redundant = True
                        for t_idx in block_triples[c_idx]:
                            if counts[t_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            # Remove it
                            sol_set.remove(c_idx)
                            blocks_in_sol.remove(c_idx)
                            for t_idx in block_triples[c_idx]:
                                counts[t_idx] -= 1
                            found_redundant = True
                            break
                    
                    if found_redundant:
                        improved = True
                        swaps_made = True
                        break # Restart main loop
                    else:
                        # Revert swap
                        sol_set.remove(new_b_idx)
                        sol_set.add(b_idx)
                        blocks_in_sol.remove(new_b_idx)
                        blocks_in_sol.append(b_idx)
                        
                        for t_idx in block_triples[new_b_idx]:
                            counts[t_idx] -= 1
                        for t_idx in block_triples[b_idx]:
                            counts[t_idx] += 1
                    
                if swaps_made:
                    break
            
            if not swaps_made and not to_remove:
                # No simple improvement
                # Maybe try a random perturbation?
                # For now, stop local search
                break
                
        return blocks_in_sol

    # Main loop
    while time.time() < deadline:
        blocks = random_greedy_cover()
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(all_blocks[b_idx]) for b_idx in blocks]
            
    return best_blocks if best_blocks is not None else []
