import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

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

    if v != 19 or k != 8 or t != 5:
        # Fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
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
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t)]
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
        
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]

    def run_greedy(seed):
        random.seed(seed)
        gain = np.full(n_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(n_t, True, dtype=bool)
        is_selected = np.full(n_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=n_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def get_block_indices(blocks):
        indices = []
        for b in blocks:
            indices.append(block_to_idx[tuple(b)])
        return indices

    def remove_single_redundancy(blocks):
        if not blocks:
            return blocks
        indices = get_block_indices(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_lists = [block_to_ts[i] for i in indices]
        
        for ts_list in block_ts_lists:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                redundant = True
                for ts_idx in block_ts_lists[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_lists[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_lists.pop(i)
                    changed = True
                    break
        return blocks

    def remove_pair_redundancy(blocks):
        if len(blocks) < 2:
            return blocks
            
        indices = get_block_indices(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_lists = [block_to_ts[i] for i in indices]
        
        for ts_list in block_ts_lists:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        n = len(blocks)
        changed = True
        while changed:
            changed = False
            # Iterate through pairs
            # To speed up, we can iterate i < j
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                for j in range(i + 1, len(blocks)):
                    # Check if removing blocks[i] and blocks[j] is safe
                    # We need to check all t-subsets in the union of blocks[i] and blocks[j]
                    # If for any ts, cover_count[ts] == (1 if ts in bi else 0) + (1 if ts in bj else 0), it's unsafe
                    # Optimization: Precompute the set of ts in bi and bj?
                    # block_ts_lists[i] and [j] are lists of indices.
                    # We can use a set for quick lookup? Or just iterate.
                    
                    # Let's create a set of ts indices for block i and j for faster check?
                    # Actually, we just need to know if cover_count[ts] > count_in_pair(ts).
                    
                    # Efficient check:
                    # 1. Iterate through ts in block_ts_lists[i]. 
                    #    If cover_count[ts] == 1 (and it's not in j, or if it is in j, cover_count[ts] == 2 and it's the only one? No)
                    #    Actually, if ts is in bi, its contribution to cover_count is at least 1.
                    #    If we remove bi, the count drops by 1.
                    #    If we also remove bj, the count drops by 1 if ts is in bj.
                    #    So, for ts in bi:
                    #       drop = 1 + (1 if ts in bj else 0)
                    #       if cover_count[ts] == drop: unsafe.
                    #    For ts in bj (but not in bi):
                    #       drop = 1
                    #       if cover_count[ts] == 1: unsafe.
                    
                    # To make this fast, we can use numpy arrays for cover_count.
                    # But the loop is in Python.
                    
                    # Let's try a simple Python loop first. If it's too slow, we optimize.
                    safe = True
                    # Check ts in bi
                    for ts_idx in block_ts_lists[i]:
                        # Determine if ts_idx is in bj
                        # We can precompute a set of ts for each block?
                        # block_ts_sets = [set(lst) for lst in block_ts_lists]
                        # This might be memory heavy but n is small.
                        pass
                    
                    # Let's precompute sets outside the loop
                    # But block_ts_lists changes when we remove blocks.
                    # So we update the list of sets.
                    
                    # For now, let's just do the check.
                    # To avoid O(56*56) per pair, we can use the fact that
                    # if cover_count[ts] > 2, it's definitely safe for that ts.
                    # Only ts with cover_count 1 or 2 are critical.
                    
                    # Let's iterate through the union of ts in bi and bj.
                    # We can use a set for bi's ts to check membership in bj quickly?
                    # Or just check cover_count.
                    
                    # Critical ts are those with low cover count.
                    # If cover_count[ts] >= 3, removing 2 blocks is safe for that ts.
                    # So we only need to check ts where cover_count[ts] <= 2.
                    
                    # Let's gather critical ts for block i and j.
                    # This is getting complex. Let's just do the straightforward check with a small optimization.
                    
                    # Optimization: 
                    # If any ts in bi has cover_count 1, and it's not in bj, it's unsafe.
                    # If any ts in bi has cover_count 2, and it IS in bj, it's unsafe.
                    # If any ts in bj has cover_count 1, and it's not in bi, it's unsafe.
                    
                    # We can precompute a set of ts indices for each block to allow O(1) membership check.
                    # Let's maintain block_ts_sets alongside block_ts_lists.
                    
                    # Since we are in a function, let's just create the sets on the fly if needed, 
                    # or better, pass them in.
                    
                    # Let's refactor to pass sets.
                    pass
        
        # The above logic is a bit messy. Let's implement a cleaner version.
        return blocks

    # Refined remove_pair_redundancy
    def remove_pair_redundancy_v2(blocks):
        if len(blocks) < 2:
            return blocks
        
        indices = get_block_indices(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_lists = [block_to_ts[i] for i in indices]
        block_ts_sets = [set(lst) for lst in block_ts_lists]
        
        for ts_list in block_ts_lists:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 2.0:
                    return blocks
                for j in range(i + 1, n):
                    # Check if removing i and j is safe
                    safe = True
                    # Check ts in block i
                    for ts_idx in block_ts_lists[i]:
                        # If ts is in block j, it contributes 2 to the removal
                        # If not, it contributes 1
                        if ts_idx in block_ts_sets[j]:
                            if cover_count[ts_idx] == 2:
                                safe = False
                                break
                        else:
                            if cover_count[ts_idx] == 1:
                                safe = False
                                break
                    if not safe:
                        continue
                    
                    # Check ts in block j that are NOT in block i
                    # We already checked ts in i.
                    # For ts in j but not i:
                    # They contribute 1 to removal.
                    # If cover_count[ts] == 1, unsafe.
                    for ts_idx in block_ts_lists[j]:
                        if ts_idx not in block_ts_sets[i]:
                            if cover_count[ts_idx] == 1:
                                safe = False
                                break
                    if not safe:
                        continue
                    
                    # Safe to remove i and j
                    # Remove j first (higher index) to keep indices valid
                    # Update cover count
                    for ts_idx in block_ts_lists[j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_ts_lists[i]:
                        cover_count[ts_idx] -= 1
                    
                    blocks.pop(j)
                    block_ts_lists.pop(j)
                    block_ts_sets.pop(j)
                    blocks.pop(i)
                    block_ts_lists.pop(i)
                    block_ts_sets.pop(i)
                    
                    changed = True
                    break # Restart outer loop
                if changed:
                    break
        return blocks

    best_blocks = None
    best_len = float('inf')
    num_restarts = 200 # Reduced to allow time for pair removal
    
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        sol = run_greedy(i)
        if sol is None:
            continue
        
        sol = remove_single_redundancy(sol)
        sol = remove_pair_redundancy_v2(sol)
        sol = remove_single_redundancy(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = [list(b) for b in sol]
            
        if best_len <= 264:
            break

    if best_blocks is None:
        best_blocks = run_greedy(0)
        if best_blocks is None:
            return []
        best_blocks = remove_single_redundancy(best_blocks)

    return best_blocks
