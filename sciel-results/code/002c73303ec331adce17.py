import itertools
import random
import time
import numpy as np

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

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings using bitsets for speed."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # For each block, list of t-subset indices it covers
    block_to_ts = [[] for _ in range(n_blocks)]
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(n_tsubs)]
    
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsi = tsub_idx[ts]
            block_to_ts[bi].append(tsi)
            ts_to_blocks[tsi].append(bi)
            
    # Convert to numpy arrays
    block_to_ts_arr = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    # Also keep as sets for fast intersection in local search
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]
    
    return tsubs, tsub_idx, blocks, block_to_ts_arr, ts_to_blocks_arr, ts_to_blocks_set, n_tsubs, n_blocks

def _run_greedy(v, k, t, seed, precomp, deadline):
    """Vectorized randomized greedy set cover."""
    tsubs, tsub_idx, blocks, block_to_ts_arr, ts_to_blocks_arr, ts_to_blocks_set, n_tsubs, n_blocks = precomp
    rng = random.Random(seed)
    
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.full(n_blocks, C(k, t), dtype=np.int32)
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected = []
    selected_set = set()
    
    while np.any(uncovered):
        if time.time() > deadline - 5.0:
            return None
        
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
            
        candidates = np.where((gain == max_gain) & (gain > 0))[0]
        if len(candidates) == 0:
            break
        best_bi = int(candidates[rng.randrange(len(candidates))])
        
        if best_bi in selected_set:
            gain[best_bi] = -1
            continue
            
        selected_set.add(best_bi)
        selected.append(best_bi)
        covered_ts = block_to_ts_arr[best_bi]
        was_uncovered_mask = uncovered[covered_ts]
        newly_covered_ts = covered_ts[was_uncovered_mask]
        
        if len(newly_covered_ts) == 0:
            gain[best_bi] = -1
            continue
            
        uncovered[newly_covered_ts] = False
        
        all_block_indices = []
        for tsi in newly_covered_ts:
            all_block_indices.extend(ts_to_blocks_arr[tsi])
            
        if all_block_indices:
            counts = np.bincount(all_block_indices, minlength=n_blocks)
            gain -= counts
            gain[selected_set] = -1000
            
        gain[best_bi] = -1000
        
    return selected

def _local_search(v, k, t, block_indices, precomp, deadline):
    """Enhanced local search with block elimination and validated swaps."""
    tsubs, tsub_idx, blocks, block_to_ts_arr, ts_to_blocks_arr, ts_to_blocks_set, n_tsubs, n_blocks = precomp
    rng = random.Random(42)
    
    sol = list(block_indices)
    sol_set = set(sol)
    
    # Build coverage counts
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
    
    # Phase 1: Iteratively remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
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
                sol_set.discard(bi)
                changed = True
                break
    
    # Phase 2: 1-exchange local search with proper validation
    max_swaps = 200
    for _ in range(max_swaps):
        if time.time() > deadline - 3.0:
            break
        if len(sol) == 0:
            break
            
        i = rng.randrange(len(sol))
        bi = sol[i]
        ts_idxs = block_to_ts_arr[bi]
        
        # Find t-subsets uniquely covered by block i
        critical_mask = cov[ts_idxs] == 1
        critical_ts = ts_idxs[critical_mask]
        
        if len(critical_ts) == 0:
            # Block is redundant, remove it
            cov[ts_idxs] -= 1
            sol.pop(i)
            sol_set.discard(bi)
            continue
        
        # Find candidate blocks that cover all critical t-subsets
        candidates = ts_to_blocks_set[int(critical_ts[0])]
        for ts_idx in critical_ts[1:]:
            candidates = candidates & ts_to_blocks_set[int(ts_idx)]
            if not candidates:
                break
        
        if not candidates:
            continue
        
        # Exclude blocks already in solution
        valid_candidates = [c for c in candidates if c not in sol_set]
        if not valid_candidates:
            continue
        
        # Try a random valid candidate
        new_bi = rng.choice(valid_candidates)
        new_ts_idxs = block_to_ts_arr[new_bi]
        
        # Tentatively apply swap
        cov[ts_idxs] -= 1
        cov[new_ts_idxs] += 1
        
        # Validate: no t-subset should have coverage < 1
        if np.min(cov) >= 1:
            sol[i] = new_bi
            sol_set.discard(bi)
            sol_set.add(new_bi)
        else:
            # Revert
            cov[new_ts_idxs] -= 1
            cov[ts_idxs] += 1
    
    # Phase 3: Another round of redundant removal after swaps
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        indices = list(range(len(sol)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2.0:
                return sol
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            if np.all(cov[ts_idxs] > 1):
                cov[ts_idxs] -= 1
                sol.pop(i)
                sol_set.discard(bi)
                changed = True
                break
    
    return sol

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy restarts
    num_restarts = 0
    while time.time() < deadline - 60.0:  # Reserve 60s for local search
        seed = num_restarts
        blocks = _run_greedy(v, k, t, seed, precomp, deadline)
        num_restarts += 1
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # If we've done a few restarts, do a quick local search on the best
                if num_restarts >= 3 and time.time() < deadline - 30.0:
                    improved = _local_search(v, k, t, best_blocks, precomp, deadline)
                    if len(improved) < best_count:
                        best_count = len(improved)
                        best_blocks = improved
        
    if best_blocks is None:
        blocks = _run_greedy(v, k, t, 0, precomp, deadline)
        if blocks is not None:
            best_blocks = blocks
        else:
            # Fallback: naive greedy
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

    # Final local search on the best solution
    if time.time() < deadline - 5.0:
        best_blocks = _local_search(v, k, t, best_blocks, precomp, deadline)
        
    tsubs, tsub_idx, blocks_list, block_to_ts_arr, ts_to_blocks_arr, ts_to_blocks_set, n_tsubs, n_blocks = precomp
    final_blocks = []
    seen = set()
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
