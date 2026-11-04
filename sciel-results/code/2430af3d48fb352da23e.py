"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 140  # 140s safety margin
    
    # Precompute all t-subsets as bitmasks for fast coverage checking
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for p in s:
            point_subsets[p].append(i)
    
    # Coverage bitset: bit i set means subset i is covered
    covered = 0
    total_mask = (1 << n_subsets) - 1 if n_subsets < 64 else None
    
    # For large n_subsets, use a set instead of bitset
    use_bitset = n_subsets <= 60
    if use_bitset:
        uncovered_count = n_subsets
    else:
        uncovered_set = set(range(n_subsets))
        uncovered_count = n_subsets
    
    blocks = []
    
    # Greedy construction
    while uncovered_count > 0:
        if time.time() > deadline - 30:
            break
        
        # Find an uncovered subset to seed a new block
        if use_bitset:
            # Find first uncovered subset
            mask = ~covered & total_mask
            first_idx = mask & (-mask)
            first_idx = first_idx.bit_length() - 1
            seed = list(all_t_subsets[first_idx])
        else:
            seed = list(all_t_subsets[next(iter(uncovered_set))])
        
        # Build block greedily
        block = seed[:]
        
        while len(block) < k:
            if time.time() > deadline - 25:
                break
            
            # For each candidate point not in block, compute how many new subsets it covers
            best_point = None
            best_gain = -1
            best_new_mask = 0
            
            for x in range(v):
                if x in block:
                    continue
                
                # Compute which subsets would be newly covered by adding x
                # A subset {s1,...,s_{t-1}, x} is newly covered if it's uncovered
                # and {s1,...,s_{t-1}} ⊆ block
                new_mask = 0
                gain = 0
                
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx[sub]
                    if use_bitset:
                        bit = 1 << idx
                        if not (covered & bit):
                            new_mask |= bit
                            gain += 1
                    else:
                        if idx in uncovered_set:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_new_mask = new_mask
            
            if best_point is None or best_gain <= 0:
                # Pick any point not in block
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
            
            block.append(best_point)
        
        if len(block) < k:
            # Fill remaining with arbitrary points
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            if use_bitset:
                if not (covered & (1 << idx)):
                    covered |= (1 << idx)
                    uncovered_count -= 1
            else:
                if idx in uncovered_set:
                    uncovered_set.discard(idx)
                    uncovered_count -= 1
        
        if uncovered_count == 0:
            break
    
    # Check if we have a valid covering
    if uncovered_count > 0:
        # Emergency: add more blocks to cover remaining subsets
        while uncovered_count > 0 and time.time() < deadline - 5:
            if use_bitset:
                mask = ~covered & total_mask
                first_idx = mask & (-mask)
                first_idx = first_idx.bit_length() - 1
                seed = list(all_t_subsets[first_idx])
            else:
                seed = list(all_t_subsets[next(iter(uncovered_set))])
            
            block = seed[:]
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if use_bitset:
                    if not (covered & (1 << idx)):
                        covered |= (1 << idx)
                        uncovered_count -= 1
                else:
                    if idx in uncovered_set:
                        uncovered_set.discard(idx)
                        uncovered_count -= 1
    
    # Local search: try to remove or swap blocks to reduce count
    if time.time() < deadline - 10:
        blocks = _local_search(blocks, v, k, t, deadline)
    
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Aggressive redundancy removal: multiple passes of removing non-critical blocks."""
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    # Precompute t-subsets for each block as list of indices
    def block_subs(block):
        return [subset_to_idx[sub] for sub in itertools.combinations(block, t)]
    
    # Build cover_count and block_subsets lists
    block_subsets = [block_subs(b) for b in blocks]
    cover_count = [0] * n_subsets
    for subs in block_subsets:
        for idx in subs:
            cover_count[idx] += 1
    
    # Multiple passes of redundancy removal
    for _pass in range(10):
        if time.time() > deadline - 8:
            break
        
        removed = set()
        for i in range(len(block_subsets)):
            if i in removed:
                continue
            
            # Check if block i is the only cover for any subset
            is_critical = False
            for idx in block_subsets[i]:
                if cover_count[idx] == 1:
                    is_critical = True
                    break
            
            if not is_critical:
                removed.add(i)
                for idx in block_subsets[i]:
                    cover_count[idx] -= 1
        
        if not removed:
            break
        
        # Rebuild lists without removed blocks
        block_subsets = [bs for i, bs in enumerate(block_subsets) if i not in removed]
        blocks = [b for i, b in enumerate(blocks) if i not in removed]
    
    # Try swap-based improvement: replace a pair of blocks with one better block
    if time.time() < deadline - 15 and len(blocks) > 2:
        blocks = _swap_improve(blocks, v, k, t, deadline, all_t_subsets, subset_to_idx)
    
    return blocks


def _swap_improve(blocks, v, k, t, deadline, all_t_subsets, subset_to_idx):
    """Try to find two blocks whose union of coverage can be replaced by one block."""
    n_subsets = len(all_t_subsets)
    
    # Precompute t-subsets for each block
    block_subsets = []
    for b in blocks:
        block_subsets.append([subset_to_idx[sub] for sub in itertools.combinations(b, t)])
    
    cover_count = [0] * n_subsets
    for subs in block_subsets:
        for idx in subs:
            cover_count[idx] += 1
    
    # For each pair of blocks, check if their union can be covered by a single new block
    n_blocks = len(blocks)
    improved = False
    
    # Limit pairs we check based on time
    max_pairs = min(n_blocks * (n_blocks - 1) // 2, 5000)
    pairs_checked = 0
    
    for i in range(n_blocks):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, n_blocks):
            if time.time() > deadline - 10:
                break
            if pairs_checked >= max_pairs:
                break
            pairs_checked += 1
            
            # Subsets covered only by block i or only by block j
            # A new block must cover all subsets that are uniquely covered by i or j
            unique_to_i = []
            unique_to_j = []
            for idx in block_subsets[i]:
                if cover_count[idx] == 1:
                    unique_to_i.append(idx)
            for idx in block_subsets[j]:
                if cover_count[idx] == 1:
                    unique_to_j.append(idx)
            
            if not unique_to_i and not unique_to_j:
                continue
            
            # The new block must contain all points from unique subsets
            required_points = set()
            for idx in unique_to_i + unique_to_j:
                for p in all_t_subsets[idx]:
                    required_points.add(p)
            
            if len(required_points) > k:
                continue
            
            # Build a candidate block containing required_points
            candidate = list(required_points)
            if len(candidate) < k:
                # Add points to maximize coverage of remaining unique subsets
                remaining = [p for p in range(v) if p not in required_points]
                # Try each remaining point to see which maximizes coverage
                best_point = None
                best_gain = 0
                for p in remaining:
                    gain = 0
                    test_block = set(candidate) | {p}
                    for idx in unique_to_i + unique_to_j:
                        if all(pt in test_block for pt in all_t_subsets[idx]):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point is not None:
                    candidate.append(best_point)
                
                # Fill rest arbitrarily
                if len(candidate) < k:
                    for p in range(v):
                        if len(candidate) >= k:
                            break
                        if p not in candidate:
                            candidate.append(p)
            
            if len(candidate) != k:
                continue
            
            candidate.sort()
            
            # Check if this candidate block actually covers all unique subsets
            cand_set = set(candidate)
            valid = True
            for idx in unique_to_i + unique_to_j:
                if not all(pt in cand_set for pt in all_t_subsets[idx]):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if removing i and j and adding candidate still covers everything
            # (it should by construction, but verify no new uncovered subsets)
            new_cover_count = cover_count[:]
            for idx in block_subsets[i]:
                new_cover_count[idx] -= 1
            for idx in block_subsets[j]:
                new_cover_count[idx] -= 1
            
            new_block_subs = [subset_to_idx[sub] for sub in itertools.combinations(candidate, t)]
            for idx in new_block_subs:
                new_cover_count[idx] += 1
            
            # Check all subsets still covered
            if any(c == 0 for c in new_cover_count):
                continue
            
            # Apply the swap
            blocks.pop(j)  # Remove j first (higher index)
            blocks.pop(i)  # Remove i
            blocks.append(candidate)
            improved = True
            break
        
        if improved:
            break
    
    return blocks
# EVOLVE-BLOCK-END
