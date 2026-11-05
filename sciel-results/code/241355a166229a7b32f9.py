import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precomputation
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris
        
    # Convert to numpy arrays for vectorized operations
    # block_triple_indices is a list of lists, convert to a flat array with offsets
    # But for simplicity and speed in Python, we might stick to lists if numpy overhead is high for small sizes
    # However, num_blocks is ~17000, num_triples is 2024.
    # Let's use a matrix representation: block_triple_matrix [num_blocks, max_triples_per_block]
    # max_triples_per_block = C(7,3) = 35
    max_tris = 35
    block_triple_matrix = np.full((num_blocks, max_tris), -1, dtype=np.int32)
    block_triple_lens = np.zeros(num_blocks, dtype=np.int32)
    
    for i, tris in enumerate(block_triple_indices):
        block_triple_matrix[i, :len(tris)] = tris
        block_triple_lens[i] = len(tris)
        
    # triple_to_blocks is a list of lists
    # For faster lookup in local search, we might want arrays here too, but let's see.
    
    best_sol = None
    best_energy = float('inf')
    
    seed_counter = 0
    
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # 1. Vectorized Greedy Construction
        sol = _greedy_cover_vectorized(rng, num_blocks, block_triple_matrix, block_triple_lens, num_triples, deadline - 5)
        
        # 2. Remove Redundant Blocks
        sol = _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples)
        
        # 3. Local Search (SA)
        time_left_for_sa = deadline - 5 - time.time()
        if time_left_for_sa < 1:
            break
            
        sol = _local_search(sol, num_blocks, block_triple_matrix, block_triple_lens, triple_to_blocks, num_triples, time_left_for_sa, rng)
        
        # Evaluate
        if sol:
            # Check validity and count
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                tris = block_triple_matrix[b_idx]
                valid_tris = tris[tris != -1]
                if len(valid_tris) > 0:
                    covered[valid_tris] = True
            
            uncovered_count = np.sum(~covered)
            
            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)

    if best_sol is None:
        return []
        
    # Convert indices back to point lists
    return [list(all_blocks[i]) for i in best_sol]

def _greedy_cover_vectorized(rng, num_blocks, block_triple_matrix, block_triple_lens, num_triples, deadline):
    uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    selected_set = set()
    remaining_blocks = np.arange(num_blocks)
    
    while np.any(uncovered) and time.time() < deadline:
        # Calculate gain for all remaining blocks
        # gain[i] = number of uncovered triples in block i
        # block_triple_matrix is [num_blocks, 35]
        # We need to sum uncovered[valid_triples] for each block
        
        # Flatten valid triples? No, just use matrix operations.
        # Create a mask for valid entries
        mask = block_triple_matrix != -1
        # Replace -1 with 0 to avoid index errors, then mask them out in sum
        safe_matrix = np.where(mask, block_triple_matrix, 0)
        
        # Get uncovered status for all positions in safe_matrix
        # uncovered is [num_triples], safe_matrix is [num_blocks, 35]
        # We want to sum uncovered[safe_matrix[i,j]] for all j where mask[i,j] is True
        
        # This is a fancy indexing operation
        # indices = safe_matrix # shape [num_blocks, 35]
        # vals = uncovered[indices] # shape [num_blocks, 35]
        # vals = np.where(mask, vals, 0)
        # gains = vals.sum(axis=1)
        
        # But doing this for all 17000 blocks every iteration is expensive (17000 * 35 * 2024? No, 17000*35 lookups).
        # 17000 * 35 = 600k lookups. This is fast in numpy.
        
        vals = uncovered[safe_matrix]
        vals = np.where(mask, vals, 0)
        gains = vals.sum(axis=1)
        
        # Exclude already selected blocks
        for s in selected_set:
            gains[s] = -1
            
        best_gain = np.max(gains)
        
        if best_gain <= 0:
            break
            
        candidates = np.where(gains == best_gain)[0]
        chosen = rng.choice(candidates)
        
        selected.append(chosen)
        selected_set.add(chosen)
        
        # Update uncovered
        tris = block_triple_matrix[chosen]
        valid_tris = tris[tris != -1]
        uncovered[valid_tris] = False
        
    return selected

def _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples):
    if not sol:
        return sol
        
    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True
            
    return sol

def _local_search(initial_sol, num_blocks, block_triple_matrix, block_triple_lens, triple_to_blocks, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)
    
    # Coverage counts
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    current_uncovered = np.sum(counts == 0)
    current_energy = len(sol) + 100 * current_uncovered
    best_sol = list(sol)
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < time_limit and T > 0.01:
        move_type = rng.random()
        
        if move_type < 0.6 and sol:
            # Removal
            to_remove = rng.choice(sol)
            tris = block_triple_matrix[to_remove]
            valid_tris = tris[tris != -1]
            
            if len(valid_tris) == 0:
                continue
                
            # Check if any triple in valid_tris has count 1
            min_count_in_block = np.min(counts[valid_tris])
            
            if min_count_in_block > 1:
                # Redundant, always accept
                counts[valid_tris] -= 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                current_uncovered = np.sum(counts == 0)
                current_energy = len(sol) + 100 * current_uncovered
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                # Not redundant
                # Count how many become 0
                delta_uncovered = np.sum(counts[valid_tris] == 1)
                delta_e = 1 + 100 * delta_uncovered
                
                if rng.random() < np.exp(-delta_e / T):
                    counts[valid_tris] -= 1
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    current_uncovered += delta_uncovered
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.9 and sol:
            # Swap
            to_remove = rng.choice(sol)
            tris_remove = block_triple_matrix[to_remove]
            valid_tris_remove = tris_remove[tris_remove != -1]
            
            if len(valid_tris_remove) == 0:
                continue
                
            # Find triples that will be uncovered
            # Those are where counts == 1
            uncovered_mask = (counts[valid_tris_remove] == 1)
            uncovered_tris = valid_tris_remove[uncovered_mask]
            
            if len(uncovered_tris) == 0:
                continue # This is effectively a removal, handled above or skipped
            
            # Find a block that covers all uncovered_tris
            # For each uncovered triple, get candidate blocks
            # We need a block that is in the intersection of candidate lists for all uncovered_tris
            # And is not in sol_set
            
            candidates = None
            for t_idx in uncovered_tris:
                cand_list = triple_to_blocks[t_idx]
                # Filter out blocks already in sol
                # Convert to set for intersection? Or just check.
                # Since uncovered_tris is small (usually 1-3), we can intersect lists.
                if candidates is None:
                    candidates = set(cand_list)
                else:
                    candidates.intersection_update(cand_list)
                
                # Remove blocks in sol_set
                candidates.difference_update(sol_set)
                
                if not candidates:
                    break
            
            if candidates:
                to_add = rng.choice(list(candidates))
                
                # Perform swap
                tris_add = block_triple_matrix[to_add]
                valid_tris_add = tris_add[tris_add != -1]
                
                # Update counts
                counts[valid_tris_remove] -= 1
                counts[valid_tris_add] += 1
                
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                sol.append(to_add)
                sol_set.add(to_add)
                
                new_uncovered = np.sum(counts == 0)
                new_energy = len(sol) + 100 * new_uncovered
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Reject: Undo swap
                    counts[valid_tris_add] -= 1
                    counts[valid_tris_remove] += 1
                    
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    sol.append(to_remove)
                    sol_set.add(to_remove)

        T *= cooling
        
        if time.time() > time_limit:
            break
            
    return best_sol
