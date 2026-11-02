"""Covering design C(19,8,5) - combined greedy + local search with vectorized operations."""
import itertools
import time
import random
import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets, all k-blocks, and their relationships."""
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    
    # For each block, which t-sets does it cover?
    block_to_tsets = []
    for block in all_kblocks:
        tsets_in_block = [tset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsets.append(tsets_in_block)
    
    # For each t-set, which blocks cover it?
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, tsets_list in enumerate(block_to_tsets):
        for ts_idx in tsets_list:
            tset_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    block_to_tsets_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsets]
    tset_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tset_to_blocks]
    
    return all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks


def _greedy_cover(v, k, t, precomp, seed=0, explore_prob=0.0, deadline=None):
    """Greedy set-cover with vectorized gain computation."""
    if deadline is None:
        deadline = time.time() + 100
    
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    rng = random.Random(seed)
    
    # Initialize gain array: each block covers C(k,t) t-sets initially
    gain = np.full(num_kblocks, len(block_to_tsets[0]), dtype=np.int32)
    is_uncovered = np.full(num_tsets, True, dtype=bool)
    is_selected = np.full(num_kblocks, False, dtype=bool)
    sol = []
    
    while True:
        if time.time() > deadline:
            return None
        
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
        
        candidates = np.where((gain == max_gain) & (~is_selected))[0]
        if len(candidates) == 0:
            break
        
        if explore_prob > 0 and rng.random() < explore_prob:
            # Explore: pick from blocks with gain >= max_gain - 3
            explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
            if len(explore_cands) > 0:
                b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        else:
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        
        sol.append(list(all_kblocks[b_idx]))
        is_selected[b_idx] = True
        gain[b_idx] = -1
        
        # Identify newly covered t-subsets
        bts = block_to_tsets_np[b_idx]
        newly_covered_mask = is_uncovered[bts]
        newly_covered = bts[newly_covered_mask]
        
        if len(newly_covered) == 0:
            continue
        
        is_uncovered[newly_covered] = False
        
        # Vectorized gain update using bincount
        blocks_to_decrement = []
        for ts_idx in newly_covered:
            blocks_to_decrement.extend(tset_to_blocks[ts_idx])
        
        if blocks_to_decrement:
            counts = np.bincount(blocks_to_decrement, minlength=num_kblocks)
            gain -= counts
            gain[is_selected] = -1000
    
    return sol


def _local_search(blocks, precomp, deadline):
    """Local search to reduce block count via redundancy removal and swaps."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    blocks = [list(b) for b in blocks]
    
    # Build coverage count
    cover_count = np.zeros(num_tsets, dtype=np.int32)
    block_ts_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_ts_indices.append(block_to_tsets[idx])
        else:
            ts_list = [tset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_ts_indices.append(ts_list)
        for ts_idx in block_ts_indices[-1]:
            cover_count[ts_idx] += 1
    
    # Phase 1: Remove redundant blocks
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            redundant = True
            for ts_idx in block_ts_indices[bi]:
                if cover_count[ts_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in block_ts_indices[bi]:
                    cover_count[ts_idx] -= 1
                blocks.pop(bi)
                block_ts_indices.pop(bi)
                improved = True
                break
    
    # Phase 2: Try swaps to reduce block count
    sol_block_indices = set()
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            sol_block_indices.add(idx)
    
    rng = random.Random(42)
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            
            # Find t-sets only covered by this block
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            
            if not unique_ts:
                # This block is redundant, remove it
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved = True
                break
            
            # Find candidate blocks that cover ALL unique_ts
            candidates = set(tset_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(tset_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
            
            new_block_idx = rng.choice(valid_candidates)
            new_block = list(all_kblocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = block_to_idx.get(tuple(old_block), -1)
            
            # Check if swap is beneficial: new block should cover more unique t-sets
            new_unique_count = 0
            for ts_idx in block_to_tsets[new_block_idx]:
                if cover_count[ts_idx] == 1:
                    new_unique_count += 1
            
            if new_unique_count >= len(unique_ts):
                # Perform swap
                blocks[i] = new_block
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)
                
                # Update coverage counts
                for ts_idx in block_to_tsets[old_block_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_tsets[new_block_idx]:
                    cover_count[ts_idx] += 1
                
                block_ts_indices[i] = block_to_tsets[new_block_idx]
                improved = True
                break
        
        if not improved:
            break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputation
    precomp = _precompute(v, k, t)
    
    # Phase 1: Multiple greedy restarts
    best_solution = _greedy_cover(v, k, t, precomp, seed=0, deadline=deadline - 20)
    if best_solution is None:
        best_solution = [list(precomp[1][0])]
    best_len = len(best_solution)
    
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
        
        sol = _greedy_cover(v, k, t, precomp, seed=i, explore_prob=explore, deadline=deadline - 20)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Phase 2: Local search to reduce block count
    if time.time() < deadline - 5:
        best_solution = _local_search(best_solution, precomp, deadline - 2)
    
    return best_solution
