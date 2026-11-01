import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    block_masks = [0] * num_k_blocks
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
            mask |= (1 << idx)
        block_to_ts.append(ts_list)
        block_masks[j] = mask

    # Convert ts_to_blocks to numpy arrays for faster intersection/iteration if needed, 
    # but lists are often faster for small intersections in Python if not vectorized.
    # We will keep lists for now but use numpy for gain updates.
    
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        # Initial gain is C(k, t) for all blocks
        gains = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        current_union_mask = 0
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Find block with max gain
            max_gain = int(np.max(gains))
            if max_gain <= 0:
                break
                
            candidates = np.where((gains == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            if explore_prob > 0 and random.random() < explore_prob:
                # Explore: pick from blocks with gain >= max_gain - 3
                explore_cands = np.where((gains >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gains[b_idx] = -1
            
            # Calculate newly covered bits
            new_bits = block_masks[b_idx] & ~current_union_mask
            current_union_mask |= block_masks[b_idx]
            
            if new_bits == 0:
                continue
                
            # Update gains for all unselected blocks
            # gain[j] -= popcount(block_masks[j] & new_bits)
            # Vectorized approach: 
            # We can't easily vectorize bit_count across a list of ints in numpy directly without a loop or ufunc.
            # However, Python's int.bit_count() is fast.
            # Looping over 75k blocks in Python is ~5-10ms. 
            # 300 steps * 10ms = 3s. Acceptable.
            
            # Optimization: Only update blocks that might have decreased gain?
            # Any block sharing a bit with new_bits has decreased gain.
            # We can iterate over all unselected blocks.
            
            # To speed up, we can skip blocks that are already selected or have gain 0.
            # But checking that is also a loop.
            
            # Let's use a list comprehension or a simple loop.
            # A numpy array of ints doesn't support bit ops on arbitrary size ints (int32/64 limit).
            # So we must use Python ints for masks.
            
            # Iterate over indices of unselected blocks
            unselected_indices = np.where(~is_selected)[0]
            for idx in unselected_indices:
                # If the block shares any bits with new_bits, its gain decreases
                # We can check if (block_masks[idx] & new_bits) != 0
                # If it is 0, gain doesn't change.
                # If not, gain decreases by popcount.
                shared = block_masks[idx] & new_bits
                if shared:
                    gains[idx] -= shared.bit_count()

        return sol

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run multiple randomized greedy
    # Since greedy is faster now, we can afford more runs or better exploration
    num_runs = 200
    for i in range(1, num_runs):
        if time.time() > deadline - 20.0:
            break
        
        explore = 0.0 if i < 50 else (0.1 if i < 100 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
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
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
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
        # Process blocks with most unique t-subsets first
        unique_counts = []
        for i in range(len(blocks)):
            uc = 0
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    uc += 1
            unique_counts.append(uc)
        
        order = list(range(len(blocks)))
        order.sort(key=lambda x: -unique_counts[x])
        
        for i in order:
            if time.time() > deadline - 3.0:
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
            
            # Update cover counts
            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts_indices[i] = block_to_ts[new_block_idx]
            
            improved = True
            
        return blocks, improved

    # Run local search
    for _ in range(50):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            break

    return best_solution
