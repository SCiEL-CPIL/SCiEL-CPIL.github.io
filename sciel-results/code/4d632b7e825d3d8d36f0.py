"""Covering design C(19,8,5) - greedy + independent-set-based local search."""
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
    
    # Precompute intersection sizes between all blocks (for independent set formulation)
    # Two blocks are "redundant" if they share >= t points (i.e., their t-sets overlap completely)
    # Actually for t=5, two 8-blocks share at least 5 points iff |intersection| >= 5
    # Build a sparse representation: for each block, list of blocks it intersects with >= t points
    block_intersections = [[] for _ in range(num_kblocks)]
    # Use bitmasks for point sets
    block_bitmasks = np.zeros(num_kblocks, dtype=np.uint32)
    for i, block in enumerate(all_kblocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        block_bitmasks[i] = mask
    
    # Precompute pairwise intersection sizes using vectorized approach
    # For v=19, we can use 19-bit masks
    # Compute intersection count for all pairs using matrix ops
    # This is 75582^2 which is ~5.7 billion - too large
    # Instead, compute on-the-fly during local search
    
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
        
        bts = block_to_tsets_np[b_idx]
        newly_covered_mask = is_uncovered[bts]
        newly_covered = bts[newly_covered_mask]
        
        if len(newly_covered) == 0:
            continue
        
        is_uncovered[newly_covered] = False
        
        # Vectorized gain update using bincount
        if len(newly_covered) > 0:
            # Flatten all tset_to_blocks for newly covered t-sets
            flat_blocks = np.concatenate([tset_to_blocks_np[ts_idx] for ts_idx in newly_covered])
            counts = np.bincount(flat_blocks, minlength=num_kblocks)
            gain -= counts
            gain[is_selected] = -1000
    
    return sol


def _local_search(blocks, precomp, deadline):
    """Independent-set-based local search: remove redundant blocks and try swaps."""
    all_tsets, all_kblocks, block_to_tsets, block_to_tsets_np, \
        tset_to_blocks, tset_to_blocks_np, tset_to_idx, block_to_idx, num_tsets, num_kblocks = precomp
    
    blocks = [list(b) for b in blocks]
    
    # Build coverage count and map blocks to their t-set index lists
    cover_count = np.zeros(num_tsets, dtype=np.int32)
    block_ts_indices = []
    block_indices = []  # index into all_kblocks for each block in solution
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_ts_indices.append(block_to_tsets_np[idx])
            block_indices.append(idx)
        else:
            ts_list = np.array([tset_to_idx[ts] for ts in itertools.combinations(b, t)], dtype=np.int32)
            block_ts_indices.append(ts_list)
            block_indices.append(-1)
        cover_count[block_ts_indices[-1]] += 1
    
    # Phase 1: Iteratively remove redundant blocks (vectorized check)
    n_blocks = len(blocks)
    for _ in range(200):
        if time.time() >= deadline:
            break
        if n_blocks == 0:
            break
        
        # Find a redundant block: one where all its t-sets have cover_count > 1
        # Check each block
        removed = False
        for bi in range(n_blocks):
            if time.time() >= deadline:
                return blocks
            bts = block_ts_indices[bi]
            if np.all(cover_count[bts] > 1):
                # Remove this block
                cover_count[bts] -= 1
                blocks.pop(bi)
                block_ts_indices.pop(bi)
                block_indices.pop(bi)
                n_blocks -= 1
                removed = True
                break
        if not removed:
            break
    
    # Phase 2: Try single-block swaps to enable further removals
    sol_block_indices = set(bi for bi in block_indices if bi >= 0)
    rng = random.Random(42)
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        n_blocks = len(blocks)
        
        for i in range(n_blocks):
            if time.time() > deadline:
                return blocks
            
            # Find t-sets only covered by this block
            bts = block_ts_indices[i]
            unique_mask = (cover_count[bts] == 1)
            unique_ts = bts[unique_mask]
            
            if len(unique_ts) == 0:
                # Redundant - remove
                cover_count[bts] -= 1
                blocks.pop(i)
                block_ts_indices.pop(i)
                block_indices.pop(i)
                improved = True
                break
            
            # Find blocks that cover ALL unique t-sets
            # Use intersection of tset_to_blocks for each unique t-set
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
            
            # Pick candidate that maximizes new coverage of currently-uncovered or singly-covered t-sets
            best_cand = None
            best_score = -1
            # Limit candidates to check for speed
            check_cands = valid_candidates[:50] if len(valid_candidates) > 50 else valid_candidates
            for c in check_cands:
                cand_bts = block_to_tsets_np[c]
                # Score: how many t-sets does this block cover that are currently uncovered or singly covered
                score = int(np.sum(cover_count[cand_bts] <= 1))
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand is None or best_score < len(unique_ts):
                continue
            
            # Perform swap
            old_idx = block_indices[i]
            if old_idx >= 0:
                sol_block_indices.discard(old_idx)
            sol_block_indices.add(best_cand)
            
            cover_count[block_to_tsets_np[best_cand]] += 1
            if old_idx >= 0:
                cover_count[block_to_tsets_np[old_idx]] -= 1
            
            blocks[i] = list(all_kblocks[best_cand])
            block_ts_indices[i] = block_to_tsets_np[best_cand]
            block_indices[i] = best_cand
            improved = True
            break
        
        if not improved:
            break
    
    # Phase 3: Final redundancy removal pass
    n_blocks = len(blocks)
    for _ in range(100):
        if time.time() >= deadline:
            break
        removed = False
        for bi in range(n_blocks):
            if time.time() >= deadline:
                return blocks
            bts = block_ts_indices[bi]
            if np.all(cover_count[bts] > 1):
                cover_count[bts] -= 1
                blocks.pop(bi)
                block_ts_indices.pop(bi)
                block_indices.pop(bi)
                n_blocks -= 1
                removed = True
                break
        if not removed:
            break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputation
    precomp = _precompute(v, k, t)
    
    # Phase 1: Multiple greedy restarts
    best_solution = _greedy_cover(v, k, t, precomp, seed=0, deadline=deadline - 25)
    if best_solution is None:
        best_solution = [list(precomp[1][0])]
    best_len = len(best_solution)
    
    # Adaptive number of runs based on time remaining
    num_runs = 200
    for i in range(1, num_runs):
        if time.time() > deadline - 30.0:
            break
        
        # Vary exploration probability
        if i < 40:
            explore = 0.0
        elif i < 80:
            explore = 0.05
        elif i < 120:
            explore = 0.1
        elif i < 160:
            explore = 0.15
        else:
            explore = 0.2
        
        sol = _greedy_cover(v, k, t, precomp, seed=i, explore_prob=explore, deadline=deadline - 30)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Phase 2: Local search to reduce block count
    if time.time() < deadline - 5:
        best_solution = _local_search(best_solution, precomp, deadline - 2)
    
    return best_solution
