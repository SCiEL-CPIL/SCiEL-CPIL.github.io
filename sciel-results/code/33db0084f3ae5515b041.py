import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin
    
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each k-block
    # shape: (n_ks, C(k,t))
    t_count = 1
    for i in range(t):
        t_count *= (k - i)
        t_count //= (i + 1)
        
    sub_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            sub_indices[i, j] = idx_map[sub]
    
    # Precompute for faster operations
    # For each block, store the set of t-subsets it covers as a frozenset for quick lookup
    # Also precompute a lookup: for each t-subset, which blocks cover it
    blocks_covering = [set() for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            ts_idx = sub_indices[i, j]
            blocks_covering[ts_idx].add(i)
    
    best_solution = None
    best_score = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        selected_blocks = []
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None
            
            # Vectorized gain calculation
            gains = uncovered[sub_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            
            selected_blocks.append(best)
            subs = sub_indices[best]
            coverage_count[subs] += 1
            uncovered[subs] = False
            
        return selected_blocks

    def reduce_cover(blocks, seed=0):
        random.seed(seed)
        
        if not blocks:
            return []
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            new_blocks = []
            for idx in order:
                b = current_blocks[idx]
                subs = sub_indices[b]
                if np.all(coverage_count[subs] > 1):
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def polish(blocks, seed=0):
        random.seed(seed)
        if not blocks:
            return blocks
            
        current_blocks = list(blocks)
        block_set = set(current_blocks)
        
        # Precompute coverage counts
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in current_blocks:
            coverage_count[sub_indices[b]] += 1
            
        improved = True
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            iter_count += 1
            
            n_blocks = len(current_blocks)
            if n_blocks == 0:
                break
            
            order = list(range(n_blocks))
            random.shuffle(order)
            
            for idx in order:
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                b = current_blocks[idx]
                subs_b = sub_indices[b]
                
                # Check if b is redundant
                if np.all(coverage_count[subs_b] > 1):
                    # Remove b
                    coverage_count[subs_b] -= 1
                    current_blocks.pop(idx)
                    block_set.discard(b)
                    improved = True
                    break
                
                # Find triples uniquely covered by b
                unique_mask = (coverage_count[subs_b] == 1)
                if not np.any(unique_mask):
                    continue # This shouldn't happen if previous check passed, but safe
                    
                unique_subs = subs_b[unique_mask]
                
                # Find candidates that cover ALL unique_subs
                # A candidate c is valid if for all u in unique_subs, c is in blocks_covering[u]
                # We can check this by intersecting the sets blocks_covering[u]
                
                candidate_set = set(blocks_covering[unique_subs[0]])
                for u in unique_subs[1:]:
                    candidate_set.intersection_update(blocks_covering[u])
                    
                # Exclude current block b
                candidate_set.discard(b)
                
                if not candidate_set:
                    continue
                    
                # Choose a candidate. 
                # Heuristic: Pick randomly to allow exploration.
                best_cand = random.choice(list(candidate_set))
                
                # Perform swap
                current_blocks[idx] = best_cand
                block_set.discard(b)
                block_set.add(best_cand)
                
                # Update coverage counts
                coverage_count[subs_b] -= 1
                coverage_count[sub_indices[best_cand]] += 1
                
                improved = True
                break # Restart pass to update all states correctly
                
        # Final reduction pass
        current_blocks = reduce_cover(current_blocks, seed + 999)
        
        return current_blocks

    # Main search loop
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks = greedy_cover(seed)
        
        if blocks is None:
            break
            
        reduced = reduce_cover(blocks, seed + 100)
        polished = polish(reduced, seed + 200)
        
        if polished is not None and len(polished) < best_score:
            best_score = len(polished)
            best_solution = list(polished)
        
        restart_count += 1
        
        # If we have found a very good solution, we can try to polish it more
        if best_score <= 76:
             # Try a few more polishes on the best solution found
             if time.time() - start_time < time_limit - 20.0:
                 for _ in range(5):
                     if time.time() - start_time > time_limit - 10.0:
                         break
                     polished2 = polish(best_solution, seed + 500 + _)
                     if polished2 and len(polished2) < best_score:
                         best_score = len(polished2)
                         best_solution = list(polished2)

    if best_solution is None:
        blocks = greedy_cover(0)
        if blocks:
            best_solution = reduce_cover(blocks, 1)
            best_solution = polish(best_solution, 2)
            
    if best_solution is None:
        return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
