"""Covering design C(19,8,5) - Vectorized greedy with bitset acceleration + local search."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute all t-subsets and k-blocks
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Map k-block to index
    block_index_map = {blk: i for i, blk in enumerate(all_k_blocks)}
    
    # Precompute: for each block, which t-subsets it covers
    block_to_ts = []
    for blk in all_k_blocks:
        ts_list = []
        for ts in itertools.combinations(blk, t):
            ts_list.append(ts_index_map[ts])
        block_to_ts.append(np.array(ts_list, dtype=np.int32))
    
    # Precompute: for each t-subset, which blocks cover it
    # This is C(19,5)=11628 t-subsets, each covered by C(19-5, 8-5)=C(14,3)=364 blocks
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, blk in enumerate(all_k_blocks):
        for ts_idx in block_to_ts[b_idx]:
            ts_to_blocks[ts_idx].append(b_idx)
    for i in range(num_t_subsets):
        ts_to_blocks[i] = np.array(ts_to_blocks[i], dtype=np.int32)
    
    # Bitset for fast membership tests
    # Each t-subset as a bit in a Python int
    ts_bitsets = [0] * num_t_subsets
    for i, ts in enumerate(all_t_subsets):
        ts_bitsets[i] = 1 << i
    
    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        
        # Gain for each block: number of currently uncovered t-subsets it covers
        gain = np.full(num_k_blocks, 0, dtype=np.int32)
        is_uncovered = np.ones(num_t_subsets, dtype=bool)
        is_selected = np.zeros(num_k_blocks, dtype=bool)
        
        # Initialize gains: each block covers C(k,t) t-subsets, all uncovered initially
        gain[:] = 56  # C(8,5) = 56
        
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
                
            if explore_prob > 0 and random.random() < explore_prob:
                # Explore: pick from blocks with gain >= max_gain - 3
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            # Identify newly covered t-subsets
            bts = block_to_ts[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            is_uncovered[newly_covered] = False
            
            # Vectorized gain update using bincount
            # For each newly covered t-subset, decrement gain of all blocks that cover it
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx].tolist())
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000
        
        # Check if all t-subsets are covered
        if np.all(~is_uncovered):
            return sol
        return None

    def optimize_redundancy(blocks, deadline):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks
        
        # Compute coverage counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = block_index_map.get(tuple(b), -1)
            if idx >= 0:
                ts_list = block_to_ts[idx]
            else:
                ts_list = np.array([ts_index_map[ts] for ts in itertools.combinations(b, t)], dtype=np.int32)
            block_ts_indices.append(ts_list)
            cover_count[ts_list] += 1
        
        n = len(blocks)
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            # Try removing in random order
            order = list(range(n))
            random.shuffle(order)
            for bi in order:
                if bi >= len(blocks):
                    continue
                ts_list = block_ts_indices[bi]
                # Check if all t-subsets in this block are covered by other blocks
                if np.all(cover_count[ts_list] > 1):
                    cover_count[ts_list] -= 1
                    blocks.pop(bi)
                    block_ts_indices.pop(bi)
                    n -= 1
                    changed = True
        
        return blocks

    def local_search(blocks, deadline):
        """Try to swap blocks to reduce count, or remove redundant ones."""
        if len(blocks) < 2:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = block_index_map.get(tuple(b), -1)
            if idx >= 0:
                ts_list = block_to_ts[idx]
            else:
                ts_list = np.array([ts_index_map[ts] for ts in itertools.combinations(b, t)], dtype=np.int33)
            block_ts_indices.append(ts_list)
            cover_count[ts_list] += 1
        
        sol_block_indices = set()
        for b in blocks:
            idx = block_index_map.get(tuple(b), -1)
            if idx >= 0:
                sol_block_indices.add(idx)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            n = len(blocks)
            order = list(range(n))
            random.shuffle(order)
            
            for bi in order:
                if bi >= len(blocks):
                    continue
                ts_list = block_ts_indices[bi]
                
                # Find t-subsets only covered by this block
                unique_mask = (cover_count[ts_list] == 1)
                unique_ts = ts_list[unique_mask]
                
                if len(unique_ts) == 0:
                    # This block is redundant
                    cover_count[ts_list] -= 1
                    blocks.pop(bi)
                    block_ts_indices.pop(bi)
                    idx = block_index_map.get(tuple(blocks[bi]) if bi < len(blocks) else tuple(blocks[-1]), -1)
                    # Just remove from set
                    improved = True
                    break
                
                # Find candidate blocks that cover all unique_ts
                if len(unique_ts) == 1:
                    candidates = ts_to_blocks[unique_ts[0]]
                else:
                    # Intersection of blocks covering each unique t-subset
                    cand_set = set(ts_to_blocks[unique_ts[0]].tolist())
                    for ts_idx in unique_ts[1:]:
                        cand_set.intersection_update(ts_to_blocks[ts_idx].tolist())
                        if not cand_set:
                            break
                    candidates = np.array(list(cand_set), dtype=np.int32) if cand_set else np.array([], dtype=np.int32)
                
                if len(candidates) == 0:
                    continue
                
                # Filter out already-selected blocks
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                
                # Pick a random valid candidate
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])
                
                # Swap
                old_block = blocks[bi]
                old_idx = block_index_map.get(tuple(old_block), -1)
                if old_idx >= 0:
                    sol_block_indices.discard(old_idx)
                
                # Update coverage
                cover_count[ts_list] -= 1
                new_ts_list = block_to_ts[new_block_idx]
                cover_count[new_ts_list] += 1
                
                blocks[bi] = new_block
                block_ts_indices[bi] = new_ts_list
                sol_block_indices.add(new_block_idx)
                improved = True
                break
        
        return blocks

    # Run multiple greedy restarts
    best_solution = None
    best_len = float('inf')
    
    # First run: deterministic greedy
    sol = run_greedy(0)
    if sol is not None:
        best_solution = sol
        best_len = len(sol)
    
    # Multiple randomized runs
    num_runs = 200
    for i in range(1, num_runs):
        if time.time() > deadline - 20.0:
            break
        
        # Vary exploration probability
        if i < 50:
            explore = 0.0
        elif i < 100:
            explore = 0.1
        elif i < 150:
            explore = 0.2
        else:
            explore = 0.3
        
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
    
    if best_solution is None:
        # Fallback: simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick the block that covers the most uncovered t-subsets
            best_block = None
            best_count = -1
            for blk in all_k_blocks:
                cnt = sum(1 for ts in itertools.combinations(blk, t) if ts in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_block = blk
            if best_block is None:
                break
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
        best_solution = blocks
        best_len = len(blocks)
    
    # Apply local search improvements
    if time.time() < deadline - 10.0:
        best_solution = optimize_redundancy(best_solution, deadline)
    
    if time.time() < deadline - 5.0:
        best_solution = local_search(best_solution, deadline)
    
    return best_solution
