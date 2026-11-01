import time
import random
import itertools
import numpy as np

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

    # Precompute block indices as numpy arrays for fast operations
    block_ts_arrays = []
    for j in range(num_k_blocks):
        block_ts_arrays.append(np.array(block_to_ts[j], dtype=np.int32))
        
    # Precompute ts_to_blocks as numpy arrays for fast vectorized updates
    ts_to_blocks_arrays = []
    for i in range(num_t_subsets):
        ts_to_blocks_arrays.append(np.array(ts_to_blocks[i], dtype=np.int32))

    # Precompute tuple-to-index map for fast block lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Randomly pick one
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected to avoid re-selection
            
            # Identify newly covered t-subsets
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            # Vectorized gain update
            # We need to decrement gain for all blocks that contain any of the newly_covered_ts
            # and are not selected.
            # Collect all block indices that contain any of these t-subsets
            # Using a list to accumulate, then bincount is efficient
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                # Count how many times each block appears in the list
                # This effectively subtracts the number of newly covered t-subsets in that block
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                # Only update unselected blocks
                # We can just subtract counts, selected blocks have gain=-1 and we don't care if they go lower
                gain -= counts
                # Ensure selected blocks stay at -1 (or lower, doesn't matter as long as they are not max)
                # But to be safe, reset selected gains to a very low number
                gain[is_selected] = -1000

        return sol

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        # Fallback if greedy timed out or failed
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run randomized greedy multiple times to find a better starting point
    num_runs = 200 # Increased runs due to faster greedy
    for i in range(1, num_runs):
        if time.time() > deadline - 10.0:
            break
            
        sol = run_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
        # Build cover_count using precomputed block_to_ts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Map solution blocks to their precomputed ts indices
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                # Fallback: compute on the fly
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
                    
        return blocks

    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Map solution blocks to their precomputed ts indices
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        # Process blocks in order of most unique coverage first (likely to be harder to replace)
        order = list(range(len(blocks)))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 2.0:
                return blocks, improved
            
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
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
                
            # Score candidates by how many additional t-subsets they cover (beyond unique_ts)
            # Prefer blocks that overlap less with existing solution to preserve coverage
            best_candidate = None
            best_score = -1
            for c in valid_candidates[:50]:  # Limit to first 50 for speed
                overlap = 0
                for ts_idx in block_to_ts[c]:
                    if cover_count[ts_idx] > 0:
                        overlap += 1
                score = len(block_to_ts[c]) - overlap  # More new coverage is better
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if best_candidate is None:
                best_candidate = random.choice(valid_candidates)
                
            new_block_idx = best_candidate
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)
            
            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            
            # Update cover counts using precomputed arrays
            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts_indices[i] = block_to_ts[new_block_idx]
            
            improved = True
            
        return blocks, improved

    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Map solution blocks to their precomputed ts indices
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)
        
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        n = len(blocks)
        
        # Try pairs of blocks to remove; see if we can find two replacement blocks
        # that allow the solution to shrink or maintain validity while enabling further reductions
        # Use random sampling of pairs to explore more combinations
        pairs_to_try = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_to_try.append((i, j))
        random.shuffle(pairs_to_try)
        
        max_pairs_to_check = min(len(pairs_to_try), 500)  # Limit for speed
        
        for i, j in pairs_to_try[:max_pairs_to_check]:
            if time.time() > deadline - 2.0:
                return blocks, improved
            if i >= len(blocks) or j >= len(blocks) or i == j:
                continue
                
            # Collect unique t-subsets for blocks[i] and blocks[j]
            unique_ts_i = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts_i.append(ts_idx)
                    
            unique_ts_j = []
            for ts_idx in block_ts_indices[j]:
                if cover_count[ts_idx] == 1:
                    unique_ts_j.append(ts_idx)
                    
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
                    # Score by how many additional t-subsets covered
                    best_candidate = None
                    best_score = -1
                    for c in valid[:30]:
                        overlap = 0
                        for ts_idx in block_to_ts[c]:
                            if cover_count[ts_idx] > 0:
                                overlap += 1
                        score = len(block_to_ts[c]) - overlap
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                    if best_candidate is None:
                        best_candidate = random.choice(valid)
                        
                    new_block_idx = best_candidate
                    new_block = list(all_k_blocks[new_block_idx])
                    
                    # Remove both blocks, add one new block
                    # This reduces size by 1!
                    old_i = blocks[i]
                    old_j = blocks[j]
                    
                    # Update cover counts using precomputed arrays
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_ts_indices[j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_block_idx]:
                        cover_count[ts_idx] += 1
                    
                    sol_block_indices.discard(get_block_idx(old_i))
                    sol_block_indices.discard(get_block_idx(old_j))
                    sol_block_indices.add(new_block_idx)
                    
                    blocks.pop(j)
                    blocks[i] = new_block
                    block_ts_indices.pop(j)
                    block_ts_indices[i] = block_to_ts[new_block_idx]
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
                # Score both candidates
                best_i = None
                best_i_score = -1
                for c in valid_i[:30]:
                    overlap = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] > 0:
                            overlap += 1
                    score = len(block_to_ts[c]) - overlap
                    if score > best_i_score:
                        best_i_score = score
                        best_i = c
                if best_i is None:
                    best_i = random.choice(valid_i)
                    
                best_j = None
                best_j_score = -1
                for c in valid_j[:30]:
                    overlap = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] > 0:
                            overlap += 1
                    score = len(block_to_ts[c]) - overlap
                    if score > best_j_score:
                        best_j_score = score
                        best_j = c
                if best_j is None:
                    best_j = random.choice(valid_j)
                    
                new_i_idx = best_i
                new_j_idx = best_j
                new_i = list(all_k_blocks[new_i_idx])
                new_j = list(all_k_blocks[new_j_idx])
                
                old_i = blocks[i]
                old_j = blocks[j]
                
                # Update cover counts using precomputed arrays
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_ts_indices[j]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_i_idx]:
                    cover_count[ts_idx] += 1
                for ts_idx in block_to_ts[new_j_idx]:
                    cover_count[ts_idx] += 1
                    
                sol_block_indices.discard(get_block_idx(old_i))
                sol_block_indices.discard(get_block_idx(old_j))
                sol_block_indices.add(new_i_idx)
                sol_block_indices.add(new_j_idx)
                
                blocks[i] = new_i
                blocks[j] = new_j
                block_ts_indices[i] = block_to_ts[new_i_idx]
                block_ts_indices[j] = block_to_ts[new_j_idx]
                improved = True
                break
                
        return blocks, improved

    # Run local search for a few iterations
    for _ in range(80):
        if time.time() > deadline - 5.0:
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
                # Try one more pass of swap with different randomization
                best_solution, improved3 = local_search_swap(best_solution)
                if improved3:
                    best_solution = optimize_redundancy(best_solution)
                    if len(best_solution) < best_len:
                        best_len = len(best_solution)
                else:
                    break

    return best_solution
