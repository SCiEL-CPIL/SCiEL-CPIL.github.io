import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

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

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build block-to-ts and ts-to-blocks
    # Using lists of lists for simplicity, then converting to numpy if needed
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(np.array(ts_list, dtype=np.int32))

    # Convert ts_to_blocks to numpy arrays for faster iteration if needed, 
    # but sets might be better for intersection. Let's keep as lists of ints for now,
    # or convert to numpy arrays and use set intersection logic.
    # Actually, for intersection, Python sets are often faster than numpy for small sets.
    ts_to_blocks_sets = [set(lst) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed):
        rng = random.Random(seed)
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_k, False, dtype=bool)
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

            # Add some randomness to tie-breaking
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts[b_idx]
            newly_covered = bts[is_uncovered[bts]]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Update gains
            # For each newly covered ts, decrement gain of all blocks covering it
            # This is expensive if done naively.
            # Optimization: only decrement if the block is not selected.
            # We can use bincount on the indices of blocks to decrement.
            
            # Collect all block indices to decrement
            to_dec = []
            for ts_idx in newly_covered:
                to_dec.extend(ts_to_blocks[ts_idx])
            
            if to_dec:
                counts = np.bincount(to_dec, minlength=num_k)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def local_search(blocks, max_iters=1000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        # Map blocks to indices
        block_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_indices.append(idx)
            else:
                # This shouldn't happen if blocks are from all_k_blocks
                return blocks, 0

        sol_set = set(block_indices)
        
        # Compute cover count
        cover_count = np.zeros(num_t, dtype=np.int32)
        for idx in block_indices:
            np.add.at(cover_count, block_to_ts[idx], 1)

        improved = 0
        iters = 0
        
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            
            if len(blocks) == 0:
                break

            # Phase 1: Remove redundant blocks
            changed = False
            for i in range(len(blocks)):
                idx = block_indices[i]
                bts = block_to_ts[idx]
                # If all t-subsets in this block are covered by >= 2 blocks, it's redundant
                if np.all(cover_count[bts] >= 2):
                    # Remove block i
                    np.subtract.at(cover_count, bts, 1)
                    blocks.pop(i)
                    block_indices.pop(i)
                    sol_set.discard(idx)
                    improved += 1
                    changed = True
                    break # Restart loop to recalculate indices if needed, but we popped from end? No, popped from i.
                    # Actually, popping from middle shifts indices. 
                    # It's safer to restart the while loop after any change.
                    break

            if changed:
                continue

            # Phase 2: 2-for-1 exchange
            # Try to replace two blocks with one
            found_swap = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 3.0:
                        break
                    
                    idx_i = block_indices[i]
                    idx_j = block_indices[j]
                    
                    # Compute unique t-subsets for i and j
                    # unique_i = bts_i where cover_count == 1
                    # But cover_count includes i and j.
                    # We need to know if a ts is covered ONLY by i or ONLY by j or ONLY by i or j.
                    # If a ts is covered by i and k (k!=j), it's not unique to i/j pair.
                    
                    # Let's compute the set of t-subsets that become uncovered if we remove i and j.
                    # These are ts such that cover_count[ts] == 1 (only i covers it) 
                    # or cover_count[ts] == 2 (i and j cover it, no one else).
                    
                    bts_i = block_to_ts[idx_i]
                    bts_j = block_to_ts[idx_j]
                    
                    # Unique to i: cover_count == 1 and ts in bts_i
                    # Unique to j: cover_count == 1 and ts in bts_j
                    # Shared only by i,j: cover_count == 2 and ts in bts_i and ts in bts_j
                    
                    # We need a new block k that covers ALL of these.
                    
                    # Let's find the union of t-subsets that are "critical" for i and j.
                    # Critical for i: ts in bts_i where cover_count[ts] == 1
                    # Critical for j: ts in bts_j where cover_count[ts] == 1
                    # Critical for both: ts in bts_i AND bts_j where cover_count[ts] == 2
                    
                    # If we remove i and j, these ts become uncovered.
                    # So we need a block k that covers this union.
                    
                    union_ts = set()
                    
                    # Ts covered only by i
                    for ts in bts_i:
                        if cover_count[ts] == 1:
                            union_ts.add(ts)
                            
                    # Ts covered only by j
                    for ts in bts_j:
                        if cover_count[ts] == 1:
                            union_ts.add(ts)
                            
                    # Ts covered only by i and j
                    # We can optimize this by intersecting bts_i and bts_j
                    # But bts_i and bts_j are numpy arrays.
                    # Let's just iterate through bts_i again and check if in bts_j?
                    # Or use set intersection.
                    set_i = set(bts_i)
                    set_j = set(bts_j)
                    common = set_i.intersection(set_j)
                    for ts in common:
                        if cover_count[ts] == 2:
                            union_ts.add(ts)

                    if not union_ts:
                        # If union_ts is empty, it means i and j are redundant?
                        # If i is redundant, we would have removed it in Phase 1.
                        # If j is redundant, same.
                        # If both are not redundant, but their union of unique ts is empty?
                        # That implies all ts in i are covered by >=2, and all in j by >=2.
                        # So they should have been removed.
                        # So this case shouldn't happen if Phase 1 is exhaustive.
                        continue

                    # Find a block k that covers union_ts
                    # k must not be in sol_set
                    # k must cover all ts in union_ts
                    
                    # Candidate search:
                    # Start with candidates for the first ts in union_ts
                    first_ts = next(iter(union_ts))
                    candidates = ts_to_blocks_sets[first_ts].copy()
                    
                    for ts in union_ts:
                        if ts == first_ts:
                            continue
                        candidates.intersection_update(ts_to_blocks_sets[ts])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                        
                    # Filter out blocks already in solution
                    valid_candidates = [c for c in candidates if c not in sol_set]
                    
                    if not valid_candidates:
                        continue
                        
                    # Pick a candidate
                    # We can pick the one that covers the most other ts to potentially make other blocks redundant?
                    # Or just random.
                    new_idx = random.choice(valid_candidates)
                    
                    # Perform swap
                    # Remove i and j
                    # Add new_idx
                    
                    # Update cover_count
                    np.subtract.at(cover_count, bts_i, 1)
                    np.subtract.at(cover_count, bts_j, 1)
                    np.add.at(cover_count, block_to_ts[new_idx], 1)
                    
                    # Check validity (should be valid by construction, but verify min cover count >= 1)
                    if np.min(cover_count) >= 1:
                        # Update blocks list
                        # Remove j first (higher index) to keep i index valid?
                        # Or remove both and add new.
                        
                        # Let's remove j, then i.
                        blocks.pop(j)
                        block_indices.pop(j)
                        sol_set.discard(idx_j)
                        
                        blocks.pop(i)
                        block_indices.pop(i)
                        sol_set.discard(idx_i)
                        
                        blocks.append(list(all_k_blocks[new_idx]))
                        block_indices.append(new_idx)
                        sol_set.add(new_idx)
                        
                        improved += 1
                        found_swap = True
                        break # Restart local search loop
                    else:
                        # Rollback
                        np.add.at(cover_count, bts_i, 1)
                        np.add.at(cover_count, bts_j, 1)
                        np.subtract.at(cover_count, block_to_ts[new_idx], 1)
                        continue

                if found_swap:
                    break
            
            if not found_swap:
                # Try 1-exchange to escape local minima
                # Find a block i and a block k not in sol such that k covers unique_ts of i
                # This doesn't reduce size, but might enable future 2-for-1s.
                # Given time constraints, we might skip this or do it sparingly.
                
                # Let's do a simple 1-exchange:
                # Pick random i
                i = random.randint(0, len(blocks) - 1)
                idx_i = block_indices[i]
                bts_i = block_to_ts[idx_i]
                
                unique_ts = set()
                for ts in bts_i:
                    if cover_count[ts] == 1:
                        unique_ts.add(ts)
                        
                if unique_ts:
                    first_ts = next(iter(unique_ts))
                    candidates = ts_to_blocks_sets[first_ts].copy()
                    for ts in unique_ts:
                        if ts == first_ts:
                            continue
                        candidates.intersection_update(ts_to_blocks_sets[ts])
                        if not candidates:
                            break
                    
                    if candidates:
                        valid_candidates = [c for c in candidates if c not in sol_set]
                        if valid_candidates:
                            new_idx = random.choice(valid_candidates)
                            # Swap
                            np.subtract.at(cover_count, bts_i, 1)
                            np.add.at(cover_count, block_to_ts[new_idx], 1)
                            if np.min(cover_count) >= 1:
                                blocks[i] = list(all_k_blocks[new_idx])
                                block_indices[i] = new_idx
                                sol_set.discard(idx_i)
                                sol_set.add(new_idx)
                                improved += 1
                                # Don't break, continue local search
                            else:
                                np.add.at(cover_count, bts_i, 1)
                                np.subtract.at(cover_count, block_to_ts[new_idx], 1)
                
                # If no improvement in this iteration, break to avoid infinite loop?
                # Or continue for a few more iterations?
                # Let's break if no improvement found in this full pass.
                break

        return blocks, improved

    # Initial greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    best_solution, _ = local_search(best_solution, max_iters=500)
    best_len = len(best_solution)

    # Run multiple greedy + local search
    for i in range(1, 50):
        if time.time() > deadline - 50.0:
            break
        sol = run_greedy(i)
        if sol is None:
            continue
        sol, _ = local_search(sol, max_iters=300)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Iterated Local Search
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb: remove 2-4 random blocks
        perturbed = list(best_solution)
        num_remove = min(random.randint(2, 4), len(perturbed) - 1)
        indices_to_remove = set(random.sample(range(len(perturbed)), num_remove))
        
        remaining = [perturbed[i] for i in range(len(perturbed)) if i not in indices_to_remove]
        
        # Repair: add blocks to cover uncovered ts
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in remaining:
            idx = get_block_idx(b)
            if idx >= 0:
                np.add.at(cover_count, block_to_ts[idx], 1)
                
        uncovered = np.where(cover_count == 0)[0]
        for ts_idx in uncovered:
            if time.time() > deadline - 5.0:
                break
            cand_blocks = ts_to_blocks[ts_idx]
            # Pick block that covers most uncovered
            best_b = -1
            best_gain = -1
            # Sample a few candidates to save time
            sample_size = min(20, len(cand_blocks))
            for b_idx in random.sample(cand_blocks, sample_size):
                bts = block_to_ts[b_idx]
                gain = np.sum(cover_count[bts] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b >= 0:
                remaining.append(list(all_k_blocks[best_b]))
                np.add.at(cover_count, block_to_ts[best_b], 1)
                
        # Local search
        perturbed, _ = local_search(remaining, max_iters=200)
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_solution = perturbed

    return best_solution
