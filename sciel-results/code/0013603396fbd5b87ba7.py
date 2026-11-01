import time
import random
import itertools
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures:
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # State for greedy
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase (Deterministic)
    while True:
        if time.time() > deadline - 10.0:
            break
            
        max_idx = int(np.argmax(gain))
        if gain[max_idx] <= 0:
            break
            
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Function to run a randomized greedy
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    # Run randomized greedy multiple times to find a better starting point
    # Each run is fast (~50-100ms), so we can do many.
    num_runs = 100
    for i in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 1.0:
                    return blocks
                
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks

    best_solution = optimize_redundancy(best_solution)
    
    # Precompute a map from tuple(block) to its index in all_k_blocks for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                idx = ts_index_map[ts]
                if cover_count[idx] == 1:
                    unique_ts.append(idx)
                    
            if not unique_ts:
                continue
            
            # Find candidate blocks that cover ALL unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)
            
            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            
            for ts in itertools.combinations(old_block, t):
                cover_count[ts_index_map[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1
            
            improved = True
            
        return blocks, improved

    # 2-exchange local search: try to replace two blocks with two different blocks
    # that allow the solution to shrink or maintain validity while enabling further reductions
    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        n = len(blocks)
        
        # Try pairs of blocks to remove; see if we can find two replacement blocks
        # that cover all uniquely-covered t-subsets of both, and ideally cover more
        for i in range(n):
            if time.time() > deadline - 1.0:
                return blocks, improved
            for j in range(i + 1, n):
                if time.time() > deadline - 1.0:
                    return blocks, improved
                    
                # Collect unique t-subsets for blocks[i] and blocks[j]
                unique_ts_i = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts_i.append(idx)
                        
                unique_ts_j = []
                for ts in itertools.combinations(blocks[j], t):
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts_j.append(idx)
                        
                if not unique_ts_i and not unique_ts_j:
                    continue
                    
                # We need to find two blocks B', B'' not in solution such that:
                # - B' covers all unique_ts_i (or they're covered by B' and B'' together)
                # - B'' covers all unique_ts_j
                # Simpler: find one block that covers ALL unique t-subsets of both i and j
                all_unique = unique_ts_i + unique_ts_j
                if not all_unique:
                    continue
                    
                # Try to find a single block covering all unique ts from both
                candidates = set(ts_to_blocks[all_unique[0]])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if candidates:
                    valid = [c for c in candidates if c not in sol_block_indices]
                    if valid:
                        new_block_idx = random.choice(valid)
                        new_block = list(all_k_blocks[new_block_idx])
                        
                        # Remove both blocks, add one new block
                        # This reduces size by 1!
                        old_i = blocks[i]
                        old_j = blocks[j]
                        
                        for ts in itertools.combinations(old_i, t):
                            cover_count[ts_index_map[ts]] -= 1
                        for ts in itertools.combinations(old_j, t):
                            cover_count[ts_index_map[ts]] -= 1
                        for ts in itertools.combinations(new_block, t):
                            cover_count[ts_index_map[ts]] += 1
                        
                        sol_block_indices.discard(get_block_idx(old_i))
                        sol_block_indices.discard(get_block_idx(old_j))
                        sol_block_indices.add(new_block_idx)
                        
                        blocks.pop(j)
                        blocks[i] = new_block
                        improved = True
                        break
                        
                # If single block doesn't work, try two separate blocks
                # Block for i's unique ts
                if unique_ts_i:
                    cand_i = set(ts_to_blocks[unique_ts_i[0]])
                    for ts_idx in unique_ts_i[1:]:
                        cand_i.intersection_update(ts_to_blocks[ts_idx])
                        if not cand_i:
                            break
                else:
                    cand_i = set(range(num_k_blocks))
                    
                if unique_ts_j:
                    cand_j = set(ts_to_blocks[unique_ts_j[0]])
                    for ts_idx in unique_ts_j[1:]:
                        cand_j.intersection_update(ts_to_blocks[ts_idx])
                        if not cand_j:
                            break
                else:
                    cand_j = set(range(num_k_blocks))
                    
                valid_i = [c for c in cand_i if c not in sol_block_indices]
                valid_j = [c for c in cand_j if c not in sol_block_indices and c not in cand_i]
                
                if valid_i and valid_j:
                    new_i_idx = random.choice(valid_i)
                    new_j_idx = random.choice(valid_j)
                    new_i = list(all_k_blocks[new_i_idx])
                    new_j = list(all_k_blocks[new_j_idx])
                    
                    old_i = blocks[i]
                    old_j = blocks[j]
                    
                    for ts in itertools.combinations(old_i, t):
                        cover_count[ts_index_map[ts]] -= 1
                    for ts in itertools.combinations(old_j, t):
                        cover_count[ts_index_map[ts]] -= 1
                    for ts in itertools.combinations(new_i, t):
                        cover_count[ts_index_map[ts]] += 1
                    for ts in itertools.combinations(new_j, t):
                        cover_count[ts_index_map[ts]] += 1
                        
                    sol_block_indices.discard(get_block_idx(old_i))
                    sol_block_indices.discard(get_block_idx(old_j))
                    sol_block_indices.add(new_i_idx)
                    sol_block_indices.add(new_j_idx)
                    
                    blocks[i] = new_i
                    blocks[j] = new_j
                    improved = True
                    break
            if improved:
                break
                
        return blocks, improved

    # WalkSAT-style local search: flip block inclusion to minimize uncovered t-subsets
    def walksat_polish(blocks, max_iters=5000):
        if not blocks:
            return blocks
            
        # Build initial cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        best_len = len(blocks)
        best_blocks = list(blocks)
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            # Find uncovered t-subsets
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered t-subset
            ts_idx = int(uncovered[random.randint(0, len(uncovered) - 1)])
            
            # Get blocks that could cover this t-subset
            candidate_blocks = ts_to_blocks[ts_idx]
            if not candidate_blocks:
                continue
            
            # Choose between adding a new block or swapping an existing one
            if random.random() < 0.4 and sol_block_indices:
                # Swap: remove a random existing block, add a new one that covers ts_idx
                existing_list = list(sol_block_indices)
                old_idx = existing_list[random.randint(0, len(existing_list) - 1)]
                
                # Find a candidate that covers ts_idx and is not already in solution
                valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                
                new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                
                # Apply swap
                old_block = list(all_k_blocks[old_idx])
                new_block = list(all_k_blocks[new_idx])
                
                for ts in itertools.combinations(old_block, t):
                    cover_count[ts_index_map[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
                
                sol_block_indices.discard(old_idx)
                sol_block_indices.add(new_idx)
                
                # Rebuild blocks list
                blocks = [list(all_k_blocks[idx]) for idx in sol_block_indices]
                
            else:
                # Add: just add a new block that covers ts_idx
                valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                
                new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                new_block = list(all_k_blocks[new_idx])
                
                for ts in itertools.combinations(new_block, t):
                    cover_count[ts_index_map[ts]] += 1
                
                sol_block_indices.add(new_idx)
                blocks.append(new_block)
            
            # Try to remove redundant blocks
            blocks = optimize_redundancy(blocks)
            
            # Update best
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
        
        return best_blocks

    # Run local search for a few iterations
    for _ in range(30):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try 2-exchange
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # Final WalkSAT polish
    if time.time() < deadline - 5.0:
        polished = walksat_polish(best_solution, max_iters=3000)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)

    return best_solution
