import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each block
    # block_t_indices[bi] is a list of triple indices
    block_t_indices = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        block_t_indices.append([idx_map[sub] for sub in subs])
        
    # Precompute inverse: triple -> list of blocks
    triple_to_blocks = [[] for _ in range(n_ts)]
    for bi, indices in enumerate(block_t_indices):
        for ti in indices:
            triple_to_blocks[ti].append(bi)

    # 2. Greedy Construction
    def greedy(seed):
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        # gain[i] = number of uncovered triples in block i
        gain = np.array([len(indices) for indices in block_t_indices], dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() - start_time > time_limit - 10:
                break
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(best_idx)
            
            subs = block_t_indices[best_idx]
            for ti in subs:
                if not covered[ti]:
                    covered[ti] = True
                    for other_bi in triple_to_blocks[ti]:
                        gain[other_bi] -= 1
            
            if np.all(covered):
                break
        
        return blocks

    # 3. Local Search
    def local_search_remove(blocks_list):
        current = list(blocks_list)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_current = []
            for b_idx in current:
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    improved = True
                    coverage_count[subs] -= 1
                else:
                    new_current.append(b_idx)
            current = new_current
        return current

    def local_search_2for1(blocks_list, time_budget):
        """
        Try to replace 2 blocks with 1 block.
        """
        current = list(blocks_list)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Calculate coverage counts
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            n_blocks = len(current)
            found = False
            
            # Sample pairs to check
            # To be efficient, we can shuffle indices and try first few pairs
            indices = list(range(n_blocks))
            random.shuffle(indices)
            
            # Limit number of pairs checked per iteration to keep it fast
            max_pairs_checked = 1000
            pairs_checked = 0
            
            for i_idx in range(len(indices)):
                for j_idx in range(i_idx + 1, len(indices)):
                    if pairs_checked >= max_pairs_checked:
                        break
                    if time.time() > deadline:
                        break
                    
                    a = current[indices[i_idx]]
                    b = current[indices[j_idx]]
                    
                    # Get unique triples for a and b
                    subs_a = block_t_indices[a]
                    subs_b = block_t_indices[b]
                    
                    unique_a = [s for s in subs_a if coverage_count[s] == 1]
                    unique_b = [s for s in subs_b if coverage_count[s] == 1]
                    
                    if not unique_a and not unique_b:
                        # Both are redundant? Should have been caught by remove pass
                        # But maybe one is redundant if we remove the other?
                        # If we remove a, b's unique triples must be covered by others.
                        # If unique_b is empty, b is redundant.
                        # This case is handled by remove pass.
                        continue
                    
                    # Union of unique triples
                    req_set = set(unique_a) | set(unique_b)
                    
                    if not req_set:
                        continue
                    
                    # Find a block c that covers all triples in req_set
                    # Use the first triple in req_set to get candidates
                    first_tri = next(iter(req_set))
                    candidates = triple_to_blocks[first_tri]
                    
                    best_c = -1
                    for c in candidates:
                        if c == a or c == b:
                            continue
                        # Check if c covers all req_set
                        # Optimization: if len(req_set) is small, check directly
                        # We need to check if all t in req_set are in block_t_indices[c]
                        # Since req_set is small, we can just iterate
                        valid = True
                        # Convert req_set to list for iteration
                        req_list = list(req_set)
                        # To speed up, we can check intersection size
                        # But Python set intersection is fast
                        # Let's just check membership in the list of c's triples
                        # Actually, checking membership in a list of size 35 is O(35).
                        # req_set size is usually small.
                        c_subs = block_t_indices[c]
                        # For very small req_set, direct check is fine
                        for tri in req_list:
                            if tri not in c_subs:
                                valid = False
                                break
                        if valid:
                            best_c = c
                            break
                    
                    if best_c != -1:
                        # Replace a and b with c
                        # Remove a and b, add c
                        # Be careful with indices if we modify the list
                        # We are using indices into 'current'
                        # We will build a new list
                        new_current = []
                        for idx in current:
                            if idx == a or idx == b:
                                continue
                            new_current.append(idx)
                        new_current.append(best_c)
                        
                        # Verify validity? 
                        # The move is valid by construction if our logic is correct.
                        # But we should ensure we didn't break coverage.
                        # The unique triples of a and b are covered by c.
                        # Triples covered by a or b but also by others are still covered.
                        # So it is valid.
                        
                        current = new_current
                        found = True
                        break # Restart search with new state
                        
                if found:
                    break
                    break # Inner break, need outer break too
                    
            if not found:
                break
                
        return current

    # 4. Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() - start_time > time_limit - 15:
            break
            
        # Greedy
        greedy_blocks = greedy(seed)
        
        # Refine
        refined = local_search_remove(greedy_blocks)
        
        # 2-for-1 swaps
        if time.time() - start_time < time_limit - 10:
             remaining = time_limit - (time.time() - start_time) - 10
             if remaining > 5.0:
                 refined = local_search_2for1(refined, min(5.0, remaining * 0.5))
             # Final removal
             refined = local_search_remove(refined)
        
        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined
            # Early stop if we hit lower bound? 
            # If we hit 69, we can't go lower. 
            # But maybe we can find 68?
            if best_count <= 69:
                # Keep searching for a bit to see if 68 is possible?
                # Or stop to save time?
                # Let's stop if we find 69, as 68 is unlikely.
                # But if we have time, try a few more seeds.
                pass

    if best_blocks is None:
        # Fallback
        return []

    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
