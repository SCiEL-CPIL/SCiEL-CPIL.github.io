import time
import numpy as np
import random
import itertools

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs) using uint8
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1
    
    # Precompute for each block: which tsubs it covers (as a set for fast lookup)
    block_tsubs = [set() for _ in range(num_kblocks)]
    for i in range(num_kblocks):
        row = M[i]
        covered = np.where(row == 1)[0]
        block_tsubs[i] = set(covered.tolist())

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    def greedy_construction(randomize=False, seed=None):
        """Greedy set cover with optional randomization for diversity."""
        local_rng = random.Random(seed) if seed is not None else rng
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
            gains = M @ uncovered
            if gains.max() == 0:
                break
            
            if randomize:
                # Pick among top-k candidates randomly
                top_k = min(10, num_kblocks)
                top_indices = np.argsort(gains)[-top_k:]
                # Weight by gain, sample with replacement
                weights = gains[top_indices].astype(float)
                weights = weights / weights.sum()
                best_idx = top_indices[local_rng.choices(range(top_k), weights=weights)[0]]
            else:
                best_idx = np.argmax(gains)
            
            selected_indices.append(best_idx)
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            
        return selected_indices

    def remove_redundant(indices):
        """Remove redundant blocks iteratively."""
        indices_set = set(indices)
        if not indices_set:
            return []
        
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in list(indices_set):
                covered = block_tsubs[idx]
                if not covered:
                    to_remove.append(idx)
                    continue
                if all(counts[c] > 1 for c in covered):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                changed = True
                
        return list(indices_set)

    def local_search(indices, max_time=None):
        """Iterated local search with 1-move and 2-move swaps."""
        if max_time is None:
            max_time = deadline - 5.0
        
        indices_set = set(indices)
        if not indices_set:
            return []
        
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
        
        # Precompute critical tsubs for each block (tsubs with count==1 that this block covers)
        def get_critical(idx):
            covered = block_tsubs[idx]
            return [c for c in covered if counts[c] == 1]
        
        improved = True
        while improved:
            improved = False
            
            # Pass 1: Remove redundant blocks
            to_remove = []
            for idx in list(indices_set):
                covered = block_tsubs[idx]
                if not covered:
                    to_remove.append(idx)
                    continue
                if all(counts[c] > 1 for c in covered):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                improved = True
            
            if not improved:
                # Pass 2: Try 1-move swaps
                idx_list = list(indices_set)
                rng.shuffle(idx_list)
                
                for b_idx in idx_list:
                    if time.monotonic() > max_time:
                        return list(indices_set)
                    
                    critical = get_critical(b_idx)
                    if not critical:
                        # Redundant
                        indices_set.remove(b_idx)
                        counts -= M[b_idx]
                        improved = True
                        break
                    
                    if len(critical) > 15:
                        continue
                    
                    # Find candidates that cover all critical tsubs
                    candidates_mask = M[:, critical].all(axis=1)
                    candidates = np.where(candidates_mask)[0]
                    candidates = [c for c in candidates if c not in indices_set]
                    
                    if not candidates:
                        continue
                    
                    # Score: maximize number of tsubs with count==1 that candidate covers
                    best_candidate = None
                    best_score = -1
                    sample_size = min(20, len(candidates))
                    sampled = rng.sample(candidates, sample_size)
                    
                    for cand in sampled:
                        cand_covered = block_tsubs[cand]
                        help_count = sum(1 for c in cand_covered if counts[c] == 1)
                        if help_count > best_score:
                            best_score = help_count
                            best_candidate = cand
                    
                    if best_candidate is not None and best_score >= 0:
                        indices_set.remove(b_idx)
                        counts -= M[b_idx]
                        indices_set.add(best_candidate)
                        counts += M[best_candidate]
                        improved = True
                        break
        
        return list(indices_set)

    def two_move_local_search(indices, max_time=None):
        """Try 2-move swaps: remove two blocks, add one or two new blocks."""
        if max_time is None:
            max_time = deadline - 5.0
        
        indices_set = set(indices)
        if len(indices_set) < 2:
            return list(indices_set)
        
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
        
        idx_list = list(indices_set)
        rng.shuffle(idx_list)
        
        improved = False
        for i in range(min(len(idx_list), 50)):
            if time.monotonic() > max_time:
                break
            b1 = idx_list[i]
            for j in range(i+1, min(i+20, len(idx_list))):
                if time.monotonic() > max_time:
                    break
                b2 = idx_list[j]
                
                # Critical tsubs after removing both b1 and b2
                crit_mask = (counts == 1) & (M[b1] | M[b2])
                critical = np.where(crit_mask)[0]
                
                if len(critical) > 20:
                    continue
                
                # Find candidate that covers all critical
                if len(critical) == 0:
                    # Both are redundant
                    indices_set.remove(b1)
                    indices_set.remove(b2)
                    counts -= M[b1]
                    counts -= M[b2]
                    improved = True
                    break
                
                candidates_mask = M[:, critical].all(axis=1)
                candidates = np.where(candidates_mask)[0]
                candidates = [c for c in candidates if c not in indices_set]
                
                if not candidates:
                    continue
                
                best_candidate = None
                best_score = -1
                sample_size = min(15, len(candidates))
                sampled = rng.sample(candidates, sample_size)
                
                for cand in sampled:
                    cand_covered = block_tsubs[cand]
                    help_count = sum(1 for c in cand_covered if counts[c] <= 2)
                    if help_count > best_score:
                        best_score = help_count
                        best_candidate = cand
                
                if best_candidate is not None:
                    indices_set.remove(b1)
                    indices_set.remove(b2)
                    counts -= M[b1]
                    counts -= M[b2]
                    indices_set.add(best_candidate)
                    counts += M[best_candidate]
                    improved = True
                    break
        
        return list(indices_set)

    # Phase 1: Multi-start greedy to find diverse initial solutions
    initial_solutions = []
    
    # First: deterministic greedy
    sol = greedy_construction(randomize=False)
    if sol:
        sol = remove_redundant(sol)
        initial_solutions.append(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Then: randomized greedy with different seeds
    for seed in [123, 456, 789, 1000, 2000, 3000, 4000, 5000]:
        if time.monotonic() > deadline - 30.0:
            break
        sol = greedy_construction(randomize=True, seed=seed)
        if sol:
            sol = remove_redundant(sol)
            initial_solutions.append(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
    
    # Phase 2: Local search on the best solution
    if time.monotonic() < deadline - 15.0:
        polished = local_search(best_solution, max_time=deadline - 10.0)
        polished = remove_redundant(polished)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = polished
    
    # Phase 3: 2-move local search
    if time.monotonic() < deadline - 10.0:
        polished = two_move_local_search(best_solution, max_time=deadline - 7.0)
        polished = remove_redundant(polished)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = polished
    
    # Phase 4: Local search on other initial solutions if time permits
    for sol in initial_solutions[1:]:
        if time.monotonic() > deadline - 5.0:
            break
        if len(sol) >= best_len:
            continue
        polished = local_search(sol, max_time=deadline - 5.0)
        polished = remove_redundant(polished)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = polished
    
    # Final reduction
    best_solution = remove_redundant(best_solution)
    
    return get_blocks(best_solution)
