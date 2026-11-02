import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 150s limit, stop at 140s to allow cleanup
    deadline = start_time + 140.0

    # 1. Precomputation
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(all_points, k))
    num_k_blocks = len(all_k_blocks)
    
    # block_to_ts: list of lists of t-subset indices covered by block i
    block_to_ts = []
    # ts_to_blocks: list of sets of block indices covering t-subset i
    ts_to_blocks = [set() for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        ts_indices = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_indices.append(idx)
            ts_to_blocks[idx].add(b_idx)
        block_to_ts.append(ts_indices)
        
    # Convert ts_to_blocks to lists for faster iteration if needed, or keep as sets
    # For gain updates, we need to iterate over blocks covering a ts.
    ts_to_blocks_lists = [list(s) for s in ts_to_blocks]
    
    # Pre-compute block to index map for O(1) lookup
    # Note: all_k_blocks contains tuples, so this is valid
    block_to_idx = {b: i for i, b in enumerate(all_k_blocks)}
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(range(num_t_subsets))
        # gain[b] = number of uncovered ts covered by block b
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
                
            # Find block with max gain among unselected
            max_gain_val = np.max(gain[~is_selected])
            if max_gain_val <= 0:
                break
            
            # Find all indices with max gain
            candidates = np.where((gain == max_gain_val) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Pick one randomly from candidates to break symmetry
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected_indices.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected
            
            # Update uncovered and gains
            b_ts = block_to_ts[b_idx]
            newly_covered = []
            for ts_idx in b_ts:
                if ts_idx in uncovered:
                    newly_covered.append(ts_idx)
                    uncovered.discard(ts_idx)
            
            # Update gains for blocks covering newly covered ts's
            if newly_covered:
                for ts_idx in newly_covered:
                    for b in ts_to_blocks_lists[ts_idx]:
                        gain[b] -= 1
        
        # Convert indices to lists of points for return
        return [list(all_k_blocks[i]) for i in selected_indices]

    def delete_blocks(blocks):
        if not blocks:
            return blocks, np.zeros(num_t_subsets, dtype=np.int32)
            
        # Initialize cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Convert blocks (lists) to indices (tuples for lookup)
        indices = []
        for b in blocks:
            indices.append(block_to_idx[tuple(b)])
        
        for idx in indices:
            ts_arr = block_to_ts[idx]
            for ts_idx in ts_arr:
                cover_count[ts_idx] += 1
        
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in random order to break symmetry
            idx_list = list(range(len(blocks)))
            random.shuffle(idx_list)
            
            for i in idx_list:
                if time.time() > deadline - 10.0:
                    return blocks, cover_count
                
                b_idx = indices[i]
                ts_arr = block_to_ts[b_idx]
                
                # Check if any ts has count == 1
                can_remove = True
                for ts_idx in ts_arr:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for ts_idx in ts_arr:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    indices.pop(i)
                    removed_any = True
                    if time.time() > deadline - 10.0:
                        return blocks, cover_count
            
        return blocks, cover_count

    def local_search(blocks, cover_count):
        if not blocks:
            return blocks, cover_count
            
        # Reconstruct indices
        indices = [block_to_idx[tuple(b)] for b in blocks]
        
        improved = True
        max_iters = 5000 # Cap iterations
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 15.0:
                break
                
            # Pick a random block to try to swap/remove
            i = random.randint(0, len(blocks) - 1)
            b_idx = indices[i]
            ts_arr = block_to_ts[b_idx]
            
            # Find unique ts's
            unique_ts = []
            for ts_idx in ts_arr:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            
            if not unique_ts:
                # Can remove directly
                for ts_idx in ts_arr:
                    cover_count[ts_idx] -= 1
                blocks.pop(i)
                indices.pop(i)
                improved = True
                continue
            
            # Need to find a replacement block that covers all unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            current_block_indices = set(indices)
            valid_candidates = [c for c in candidates if c not in current_block_indices]
            
            if not valid_candidates:
                continue
            
            # Pick a candidate. Heuristic: pick one that covers the most "vulnerable" (low count) other ts's?
            best_cand = None
            best_score = -1
            
            # Limit candidates to avoid O(N) scan if too many
            if len(valid_candidates) > 20:
                valid_candidates = random.sample(valid_candidates, 20)
                
            for c_idx in valid_candidates:
                c_ts = block_to_ts[c_idx]
                score = 0
                for ts_idx in c_ts:
                    if cover_count[ts_idx] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c_idx
            
            if best_cand is None:
                continue
            
            # Perform swap
            # 1. Remove b_i
            for ts_idx in ts_arr:
                cover_count[ts_idx] -= 1
            
            # 2. Add best_cand
            c_ts_arr = block_to_ts[best_cand]
            for ts_idx in c_ts_arr:
                cover_count[ts_idx] += 1
            
            # Update block and index
            blocks[i] = list(all_k_blocks[best_cand])
            indices[i] = best_cand
            
            # Try to remove redundant blocks
            any_removed = False
            for j in range(len(blocks)):
                if j == i:
                    continue
                b_j_idx = indices[j]
                ts_j = block_to_ts[b_j_idx]
                can_remove = True
                for ts_idx in ts_j:
                    if cover_count[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts_idx in ts_j:
                        cover_count[ts_idx] -= 1
                    blocks.pop(j)
                    indices.pop(j)
                    any_removed = True
                    if j < i:
                        i -= 1
                    improved = True
                    break
            
            if not any_removed:
                # Revert swap
                for ts_idx in c_ts_arr:
                    cover_count[ts_idx] -= 1
                for ts_idx in ts_arr:
                    cover_count[ts_idx] += 1
                blocks[i] = list(all_k_blocks[b_idx])
                indices[i] = b_idx
        
        return blocks, cover_count

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple instances
    num_runs = 10
    for i in range(num_runs):
        if time.time() > deadline - 30.0:
            break
            
        random.seed(42 + i)
        blocks = greedy_cover(42 + i)
        
        if not blocks:
            continue
            
        blocks, cover_count = delete_blocks(blocks)
        
        # Local search
        if time.time() < deadline - 20.0:
            blocks, cover_count = local_search(blocks, cover_count)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
    if best_blocks is None:
        # Fallback
        blocks = greedy_cover(42)
        blocks, _ = delete_blocks(blocks)
        return blocks
        
    return best_blocks

# Helper function for combinatorics
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
