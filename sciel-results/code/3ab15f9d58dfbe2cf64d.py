import itertools
import random
import time
import numpy as np

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings with bitsets."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    block_idx = {b: i for i, b in enumerate(blocks)}
    
    # For each block, list of t-subset indices it covers
    block_to_ts = [[] for _ in range(n_blocks)]
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(n_tsubs)]
    
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsi = tsub_idx[ts]
            block_to_ts[bi].append(tsi)
            ts_to_blocks[tsi].append(bi)
            
    # Convert to numpy arrays for faster access in loops
    block_to_ts_arr = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    # Bitset representation for each block: a Python int where bit i is set if block covers t-subset i
    block_bitsets = [0] * n_blocks
    for bi in range(n_blocks):
        bits = 0
        for tsi in block_to_ts_arr[bi]:
            bits |= (1 << tsi)
        block_bitsets[bi] = bits
    
    return tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets

def _run_greedy(v, k, t, seed, precomp, deadline):
    """Vectorized randomized greedy set cover."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets = precomp
    rng = random.Random(seed)
    
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.full(n_blocks, C(k, t), dtype=np.int32)
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected = []
    
    while np.any(uncovered):
        if time.time() > deadline - 5.0:
            return None
        
        # Find block with max gain
        max_gain = np.max(gain)
        if max_gain <= 0:
            break
            
        # Random tie-breaking
        candidates = np.where(gain == max_gain)[0]
        best_bi = candidates[rng.randrange(len(candidates))]
        
        selected.append(best_bi)
        
        # Get t-subsets covered by this block
        covered_ts = block_to_ts_arr[best_bi]
        
        # Identify which of these were actually uncovered
        was_uncovered_mask = uncovered[covered_ts]
        newly_covered_ts = covered_ts[was_uncovered_mask]
        
        if len(newly_covered_ts) == 0:
            gain[best_bi] = -1
            continue
            
        # Update uncovered status
        uncovered[newly_covered_ts] = False
        
        # Update gains for all blocks that cover any of the newly covered t-subsets
        all_block_indices = []
        for tsi in newly_covered_ts:
            all_block_indices.extend(ts_to_blocks_arr[tsi])
            
        if all_block_indices:
            counts = np.bincount(all_block_indices, minlength=n_blocks)
            gain -= counts
            
        # Mark selected block as unavailable
        gain[best_bi] = -1
        
    return selected

def _vnd_local_search(v, k, t, block_indices, precomp, deadline):
    """Variable Neighborhood Descent with 1-swap, 2-swap, and 3-swap moves."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets = precomp
    rng = random.Random(42)
    
    sol = list(block_indices)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # Phase 1: Remove redundant blocks
        indices = list(range(len(sol)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 3.0:
                return sol
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            if np.all(cov[ts_idxs] > 1):
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break
                
        if improved:
            continue
            
        # Phase 2: Try 1-swaps
        num_swaps_tried = 0
        max_swaps = 50
        
        while num_swaps_tried < max_swaps and time.time() < deadline - 3.0:
            num_swaps_tried += 1
            if len(sol) == 0:
                break
                
            i = rng.randrange(len(sol))
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            
            critical_mask = cov[ts_idxs] == 1
            critical_ts = ts_idxs[critical_mask]
            
            if len(critical_ts) == 0:
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break
                
            crit_sorted = sorted(critical_ts, key=lambda x: len(ts_to_blocks_arr[x]))
            first_crit = crit_sorted[0]
            
            found_replacement = False
            candidates = list(ts_to_blocks_arr[first_crit])
            rng.shuffle(candidates)
            
            for bi2 in candidates:
                if bi2 == bi:
                    continue
                ts_idxs_bi2 = block_to_ts_arr[bi2]
                if np.all(np.isin(critical_ts, ts_idxs_bi2)):
                    cov[ts_idxs] -= 1
                    cov[ts_idxs_bi2] += 1
                    sol[i] = bi2
                    improved = True
                    found_replacement = True
                    break
                    
            if found_replacement:
                break
                
        if improved:
            continue
            
        # Phase 3: Try 2-swaps
        if len(sol) >= 2:
            num_2swaps_tried = 0
            max_2swaps = 15
            
            while num_2swaps_tried < max_2swaps and time.time() < deadline - 3.0:
                num_2swaps_tried += 1
                
                i1 = rng.randrange(len(sol))
                i2 = rng.randrange(len(sol))
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                    
                bi1 = sol[i1]
                bi2 = sol[i2]
                
                ts_idxs_1 = block_to_ts_arr[bi1]
                ts_idxs_2 = block_to_ts_arr[bi2]
                
                temp_cov = cov.copy()
                temp_cov[ts_idxs_1] -= 1
                temp_cov[ts_idxs_2] -= 1
                
                newly_uncovered = np.where(temp_cov == 0)[0]
                
                if len(newly_uncovered) == 0:
                    cov[ts_idxs_1] -= 1
                    cov[ts_idxs_2] -= 1
                    sol.pop(i2)
                    sol.pop(i1)
                    improved = True
                    break
                    
                newly_uncovered_sorted = sorted(newly_uncovered, key=lambda x: len(ts_to_blocks_arr[x]))
                first_nu = newly_uncovered_sorted[0]
                
                cand1 = list(ts_to_blocks_arr[first_nu])
                rng.shuffle(cand1)
                
                found_2swap = False
                for bi_new1 in cand1[:30]:
                    if bi_new1 == bi1 or bi_new1 == bi2:
                        continue
                    
                    ts_idxs_new1 = block_to_ts_arr[bi_new1]
                    remaining_uncovered = [tsi for tsi in newly_uncovered if tsi not in ts_idxs_new1]
                    
                    if len(remaining_uncovered) == 0:
                        continue
                    
                    remaining_sorted = sorted(remaining_uncovered, key=lambda x: len(ts_to_blocks_arr[x]))
                    second_nu = remaining_sorted[0]
                    
                    cand2 = list(ts_to_blocks_arr[second_nu])
                    rng.shuffle(cand2)
                    
                    for bi_new2 in cand2[:30]:
                        if bi_new2 == bi1 or bi_new2 == bi2 or bi_new2 == bi_new1:
                            continue
                        
                        ts_idxs_new2 = block_to_ts_arr[bi_new2]
                        covered_by_new = set(ts_idxs_new1) | set(ts_idxs_new2)
                        if all(tsi in covered_by_new for tsi in newly_uncovered):
                            cov[ts_idxs_1] -= 1
                            cov[ts_idxs_2] -= 1
                            cov[ts_idxs_new1] += 1
                            cov[ts_idxs_new2] += 1
                            sol[i1] = bi_new1
                            sol[i2] = bi_new2
                            improved = True
                            found_2swap = True
                            break
                            
                    if found_2swap:
                        break
                        
            if found_2swap:
                break
                
        # Phase 4: Try 3-swaps (new)
        if len(sol) >= 3:
            num_3swaps_tried = 0
            max_3swaps = 8
            
            while num_3swaps_tried < max_3swaps and time.time() < deadline - 3.0:
                num_3swaps_tried += 1
                
                # Pick three random blocks to remove
                idxs = rng.sample(range(len(sol)), 3)
                idxs.sort()
                i1, i2, i3 = idxs
                
                bi1 = sol[i1]
                bi2 = sol[i2]
                bi3 = sol[i3]
                
                ts_idxs_1 = block_to_ts_arr[bi1]
                ts_idxs_2 = block_to_ts_arr[bi2]
                ts_idxs_3 = block_to_ts_arr[bi3]
                
                temp_cov = cov.copy()
                temp_cov[ts_idxs_1] -= 1
                temp_cov[ts_idxs_2] -= 1
                temp_cov[ts_idxs_3] -= 1
                
                newly_uncovered = np.where(temp_cov == 0)[0]
                
                if len(newly_uncovered) == 0:
                    cov[ts_idxs_1] -= 1
                    cov[ts_idxs_2] -= 1
                    cov[ts_idxs_3] -= 1
                    for idx in [i3, i2, i1]:
                        sol.pop(idx)
                    improved = True
                    break
                    
                # Find 3 blocks that cover all newly_uncovered
                newly_uncovered_sorted = sorted(newly_uncovered, key=lambda x: len(ts_to_blocks_arr[x]))
                first_nu = newly_uncovered_sorted[0]
                
                cand1 = list(ts_to_blocks_arr[first_nu])
                rng.shuffle(cand1)
                
                found_3swap = False
                for bi_new1 in cand1[:25]:
                    if bi_new1 in (bi1, bi2, bi3):
                        continue
                    
                    ts_idxs_new1 = block_to_ts_arr[bi_new1]
                    remaining1 = [tsi for tsi in newly_uncovered if tsi not in ts_idxs_new1]
                    
                    if len(remaining1) == 0:
                        continue
                    
                    remaining1_sorted = sorted(remaining1, key=lambda x: len(ts_to_blocks_arr[x]))
                    second_nu = remaining1_sorted[0]
                    
                    cand2 = list(ts_to_blocks_arr[second_nu])
                    rng.shuffle(cand2)
                    
                    for bi_new2 in cand2[:25]:
                        if bi_new2 in (bi1, bi2, bi3, bi_new1):
                            continue
                        
                        ts_idxs_new2 = block_to_ts_arr[bi_new2]
                        remaining2 = [tsi for tsi in remaining1 if tsi not in ts_idxs_new2]
                        
                        if len(remaining2) == 0:
                            continue
                        
                        remaining2_sorted = sorted(remaining2, key=lambda x: len(ts_to_blocks_arr[x]))
                        third_nu = remaining2_sorted[0]
                        
                        cand3 = list(ts_to_blocks_arr[third_nu])
                        rng.shuffle(cand3)
                        
                        for bi_new3 in cand3[:25]:
                            if bi_new3 in (bi1, bi2, bi3, bi_new1, bi_new2):
                                continue
                            
                            ts_idxs_new3 = block_to_ts_arr[bi_new3]
                            covered_by_new = set(ts_idxs_new1) | set(ts_idxs_new2) | set(ts_idxs_new3)
                            if all(tsi in covered_by_new for tsi in newly_uncovered):
                                cov[ts_idxs_1] -= 1
                                cov[ts_idxs_2] -= 1
                                cov[ts_idxs_3] -= 1
                                cov[ts_idxs_new1] += 1
                                cov[ts_idxs_new2] += 1
                                cov[ts_idxs_new3] += 1
                                sol[i1] = bi_new1
                                sol[i2] = bi_new2
                                sol[i3] = bi_new3
                                improved = True
                                found_3swap = True
                                break
                                
                        if found_3swap:
                            break
                            
                    if found_3swap:
                        break
                        
                if found_3swap:
                    break
                    
    return sol

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

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy restarts
    num_restarts = 0
    while time.time() < deadline - 45.0:
        seed = num_restarts
        blocks = _run_greedy(v, k, t, seed, precomp, deadline)
        num_restarts += 1
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
    if best_blocks is None:
        blocks = _run_greedy(v, k, t, 0, precomp, deadline)
        if blocks is not None:
            best_blocks = blocks
        else:
            uncovered = set(itertools.combinations(range(v), t))
            best_blocks = []
            while uncovered and time.time() < deadline - 5.0:
                target = next(iter(uncovered))
                best_block = None
                best_gain = -1
                remaining = [x for x in range(v) if x not in target]
                need = k - t
                for comb in itertools.combinations(remaining, need):
                    block = tuple(sorted(target + comb))
                    gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                if best_block is None:
                    break
                best_blocks.append(best_block)
                uncovered.difference_update(itertools.combinations(best_block, t))
            return [list(b) for b in best_blocks]

    # VND local search on the best solution
    if time.time() < deadline - 5.0:
        best_blocks = _vnd_local_search(v, k, t, best_blocks, precomp, deadline)
        
    # Convert indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets = precomp
    final_blocks = []
    seen = set()
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
