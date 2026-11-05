import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Precompute reverse map: for each tsub, which blocks cover it
    # This is useful for finding candidates quickly
    tsub_to_blocks = []
    for j in range(num_tsubs):
        blocks_covering = np.where(M[:, j] == 1)[0]
        tsub_to_blocks.append(blocks_covering)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Helper: Greedy construction
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
            # Gains: number of uncovered tsubs covered by each block
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(int(best_idx))
            # Update uncovered
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            
        return selected_indices

    # Helper: Remove redundant blocks
    def remove_redundant(indices):
        indices_set = set(indices)
        # Calculate counts
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in indices_set:
                row = M[idx]
                covered = np.where(row == 1)[0]
                # If all covered tsubs have count > 1, it's redundant
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                changed = True
                
        return list(indices_set)

    # Helper: Try to swap blocks to reduce count - optimized version
    def local_search(indices, max_iterations=100):
        indices_set = set(indices)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        for idx in indices_set:
            selected_mask[idx] = True
            
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            if time.monotonic() > deadline - 5.0:
                break
                
            # First pass: remove redundant
            new_indices = remove_redundant(list(indices_set))
            if len(new_indices) < len(indices_set):
                # Update structures
                removed = indices_set - set(new_indices)
                for idx in removed:
                    selected_mask[idx] = False
                for idx in new_indices:
                    selected_mask[idx] = True
                indices_set = set(new_indices)
                counts = np.zeros(num_tsubs, dtype=np.int32)
                for idx in indices_set:
                    counts += M[idx]
                improved = True
                continue
            
            # Second pass: try swaps
            idx_list = list(indices_set)
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.monotonic() > deadline - 5.0:
                    return list(indices_set)
                
                # Critical tsubs: those covered ONLY by b_idx
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                critical_mask = counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]
                
                if len(critical_indices) == 0:
                    # Redundant, should have been removed
                    indices_set.remove(b_idx)
                    selected_mask[b_idx] = False
                    counts -= row
                    improved = True
                    break
                
                # Find candidates that cover all critical_indices and are not selected
                # Use precomputed reverse map for efficiency
                if len(critical_indices) == 1:
                    candidates = tsub_to_blocks[critical_indices[0]]
                    candidates = candidates[~selected_mask[candidates]]
                elif len(critical_indices) <= 3:
                    # Intersection of block sets for each critical tsub
                    candidate_set = set(tsub_to_blocks[critical_indices[0]])
                    for ci in critical_indices[1:]:
                        candidate_set &= set(tsub_to_blocks[ci])
                    candidates = np.array(list(candidate_set), dtype=np.int64)
                    if len(candidates) > 0:
                        candidates = candidates[~selected_mask[candidates]]
                    else:
                        candidates = np.array([], dtype=np.int64)
                else:
                    # Too many constraints, use numpy
                    candidates_mask = M[:, critical_indices].all(axis=1)
                    candidates_mask &= ~selected_mask
                    candidates = np.where(candidates_mask)[0]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    c_idx = int(candidates[rng.randrange(len(candidates))])
                    
                    # Perform swap
                    indices_set.remove(b_idx)
                    selected_mask[b_idx] = False
                    counts -= M[b_idx]
                    indices_set.add(c_idx)
                    selected_mask[c_idx] = True
                    counts += M[c_idx]
                    improved = True
                    break
                    
        return list(indices_set)

    # Phase 1: Initial Greedy
    initial_sol = greedy_construction()
    if initial_sol:
        best_solution = initial_sol
        best_len = len(initial_sol)

    # Phase 2: ILS with optimized local search
    current_sol = best_solution[:]
    
    # First do a thorough local search on the initial solution
    polished = local_search(current_sol, max_iterations=200)
    if len(polished) < best_len:
        best_len = len(polished)
        best_solution = polished[:]
        current_sol = polished[:]

    for _ in range(500):  # Max iterations, bounded by time
        if time.monotonic() > deadline - 10.0:
            break
            
        # Local Search
        polished_sol = local_search(current_sol, max_iterations=50)
        
        if len(polished_sol) < best_len:
            best_len = len(polished_sol)
            best_solution = polished_sol[:]
            current_sol = polished_sol[:]
        else:
            current_sol = polished_sol[:]
            
        # Perturbation - less destructive
        n_remove = min(3 + rng.randint(0, 3), len(current_sol))
        if n_remove > 0:
            to_remove = rng.sample(current_sol, n_remove)
            current_sol_set = set(current_sol)
            for b in to_remove:
                current_sol_set.remove(b)
            current_sol = list(current_sol_set)
            
            # Greedy repair
            counts = np.zeros(num_tsubs, dtype=np.int32)
            for idx in current_sol:
                counts += M[idx]
                
            uncovered_mask = counts == 0
            
            while np.any(uncovered_mask):
                if time.monotonic() > deadline - 5.0:
                    break
                gains = M @ uncovered_mask
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                current_sol.append(best_idx)
                counts += M[best_idx]
                uncovered_mask = counts == 0
                
            # If repair failed or got too big, revert to best
            if len(current_sol) > best_len + 5:
                current_sol = best_solution[:]

    # Phase 3: Final high-precision polish pass
    if best_solution is not None:
        if time.monotonic() < deadline - 15.0:
            final_polish = local_search(best_solution, max_iterations=500)
            if len(final_polish) < best_len:
                best_len = len(final_polish)
                best_solution = final_polish[:]
                
        # One more pass with even more iterations if time permits
        if time.monotonic() < deadline - 5.0:
            final_polish2 = local_search(best_solution, max_iterations=1000)
            if len(final_polish2) < best_len:
                best_len = len(final_polish2)
                best_solution = final_polish2[:]

    if best_solution is None:
        return []
        
    return get_blocks(best_solution)
