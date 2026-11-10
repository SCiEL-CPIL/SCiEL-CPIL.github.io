"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _build_sub_index(v, t):
    """Build all t-subsets and index mapping."""
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    return all_subs, sub_to_idx, len(all_subs)


def _block_to_sub_indices(block, sub_to_idx):
    """Return numpy array of t-subset indices covered by this block."""
    subs = list(itertools.combinations(block, t))
    if not subs:
        return np.array([], dtype=np.int32)
    return np.array([sub_to_idx[s] for s in subs], dtype=np.int32)


def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering: pick first uncovered t-subset, grow a k-block
    maximizing newly-covered t-subsets at each step.
    """
    all_subs, sub_to_idx, n_subs = _build_sub_index(v, t)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []
    
    # Precompute: for each point, which t-subsets contain it
    point_sub_indices = [[] for _ in range(v)]
    for idx, sub in enumerate(all_subs):
        for x in sub:
            point_sub_indices[x].append(idx)
    
    while not covered.all():
        if time.time() > deadline:
            break
        
        # Find first uncovered subset
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_sub = all_subs[start_idx]
        
        # Build block greedily
        block = list(start_sub)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                
            # Compute gain: how many uncovered t-subsets become covered
            # A t-subset is covered if all its elements are in the block
            # Adding x covers t-subsets that contain x and (t-1) elements from block
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                candidate = tuple(sorted(rest + (x,)))
                idx = sub_to_idx.get(candidate, -1)
                if idx >= 0 and not covered[idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, deadline):
    """Remove blocks that don't uniquely cover any t-subset. Iterates to fixed point."""
    if len(blocks) <= 1:
        return blocks
    
    all_subs, sub_to_idx, n_subs = _build_sub_index(v, t)
    
    # Precompute sub indices for each block
    block_sub_idx = [_block_to_sub_indices(b, sub_to_idx) for b in blocks]
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for bsi in block_sub_idx:
        if len(bsi) > 0:
            np.add.at(coverage, bsi, 1)
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_sub_idx = []
        new_coverage = coverage.copy()
        
        for i, block in enumerate(blocks):
            bsi = block_sub_idx[i]
            has_unique = False
            if len(bsi) > 0:
                has_unique = np.any(coverage[bsi] == 1)
            
            if not has_unique:
                if len(bsi) > 0:
                    np.add.at(new_coverage, bsi, -1)
                changed = True
            else:
                new_blocks.append(block)
                new_block_sub_idx.append(bsi)
            
            if time.time() > deadline:
                return new_blocks + blocks[i + 1:]
        
        blocks = new_blocks
        block_sub_idx = new_block_sub_idx
        coverage = new_coverage
    
    return blocks


def _local_search(blocks, v, k, t, deadline, seed=42):
    """
    Local search: remove redundant blocks, swap blocks for better ones.
    Uses precomputed block subset indices for fast coverage updates.
    """
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    all_subs, sub_to_idx, n_subs = _build_sub_index(v, t)
    
    # Precompute block sub indices
    def compute_block_subs(block):
        return _block_to_sub_indices(block, sub_to_idx)
    
    block_subs = [compute_block_subs(b) for b in current_blocks]
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for bsi in block_subs:
        if len(bsi) > 0:
            np.add.at(coverage, bsi, 1)
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.001
    
    while time.time() < deadline and temp > min_temp:
        move_type = rng.randint(0, 2)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            bsi = block_subs[idx]
            
            can_remove = True
            if len(bsi) > 0:
                can_remove = np.all(coverage[bsi] > 1)
            
            if can_remove:
                if len(bsi) > 0:
                    np.add.at(coverage, bsi, -1)
                current_blocks.pop(idx)
                block_subs.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
        
        elif move_type == 1 and len(current_blocks) > 1:
            # Swap: replace a block with a better one
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_bsi = block_subs[idx]
            
            can_remove = True
            if len(old_bsi) > 0:
                can_remove = np.all(coverage[old_bsi] > 1)
            
            if not can_remove:
                temp *= cooling
                continue
            
            # Temporarily remove
            if len(old_bsi) > 0:
                np.add.at(coverage, old_bsi, -1)
            
            # Find a good replacement block
            best_block = None
            best_gain = -1
            
            # Try several random blocks
            for _ in range(5):
                candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                c_bsi = compute_block_subs(candidate)
                if len(c_bsi) > 0:
                    gain = np.sum(coverage[c_bsi] == 0)
                else:
                    gain = 0
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate
            
            if best_block is not None and best_gain > 0:
                new_bsi = compute_block_subs(best_block)
                if len(new_bsi) > 0:
                    np.add.at(coverage, new_bsi, 1)
                current_blocks[idx] = best_block
                block_subs[idx] = new_bsi
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            else:
                # Roll back
                if len(old_bsi) > 0:
                    np.add.at(coverage, old_bsi, 1)
            
            temp *= cooling
            continue
        
        else:
            # Perturb: modify a random block
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_bsi = block_subs[idx]
            
            # Create a new block by changing 1-2 elements
            new_block = list(old_block)
            num_changes = rng.randint(1, min(2, k))
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val
            
            new_block.sort()
            
            if new_block == old_block:
                temp *= cooling
                continue
            
            new_bsi = compute_block_subs(new_block)
            
            # Check if replacing old_block with new_block maintains coverage
            valid = True
            if len(old_bsi) > 0:
                # For each t-subset in old_block that's uniquely covered,
                # it must also be in new_block
                unique_mask = coverage[old_bsi] == 1
                if np.any(unique_mask):
                    unique_subs = old_bsi[unique_mask]
                    new_set = set(new_bsi.tolist()) if len(new_bsi) > 0 else set()
                    for us in unique_subs:
                        if us not in new_set:
                            valid = False
                            break
            
            if valid:
                if len(old_bsi) > 0:
                    np.add.at(coverage, old_bsi, -1)
                if len(new_bsi) > 0:
                    np.add.at(coverage, new_bsi, 1)
                current_blocks[idx] = new_block
                block_subs[idx] = new_bsi
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            
            temp *= cooling
    
    return best_blocks


def _sa_search(blocks, v, k, t, deadline, seed=123):
    """
    Simulated annealing with aggressive exploration.
    Moves: remove redundant, swap, perturb, add block.
    """
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    all_subs, sub_to_idx, n_subs = _build_sub_index(v, t)
    
    def compute_block_subs(block):
        return _block_to_sub_indices(block, sub_to_idx)
    
    block_subs = [compute_block_subs(b) for b in current_blocks]
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for bsi in block_subs:
        if len(bsi) > 0:
            np.add.at(coverage, bsi, 1)
    
    temp = 5.0
    cooling = 0.995
    min_temp = 0.01
    max_iter = 5000
    
    iteration = 0
    while time.time() < deadline and temp > min_temp and iteration < max_iter:
        iteration += 1
        move_type = rng.randint(0, 3)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            bsi = block_subs[idx]
            
            can_remove = True
            if len(bsi) > 0:
                can_remove = np.all(coverage[bsi] > 1)
            
            if can_remove:
                if len(bsi) > 0:
                    np.add.at(coverage, bsi, -1)
                current_blocks.pop(idx)
                block_subs.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
        
        elif move_type == 1 and len(current_blocks) > 1:
            # Swap: replace a block with a better one
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_bsi = block_subs[idx]
            
            can_remove = True
            if len(old_bsi) > 0:
                can_remove = np.all(coverage[old_bsi] > 1)
            
            if not can_remove:
                temp *= cooling
                continue
            
            if len(old_bsi) > 0:
                np.add.at(coverage, old_bsi, -1)
            
            best_block = None
            best_gain = -1
            
            for _ in range(10):
                candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                c_bsi = compute_block_subs(candidate)
                if len(c_bsi) > 0:
                    gain = np.sum(coverage[c_bsi] == 0)
                else:
                    gain = 0
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate
            
            if best_block is not None and best_gain > 0:
                new_bsi = compute_block_subs(best_block)
                if len(new_bsi) > 0:
                    np.add.at(coverage, new_bsi, 1)
                current_blocks[idx] = best_block
                block_subs[idx] = new_bsi
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            else:
                if len(old_bsi) > 0:
                    np.add.at(coverage, old_bsi, 1)
            
            temp *= cooling
            continue
        
        elif move_type == 2:
            # Perturb: modify a random block more aggressively
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_bsi = block_subs[idx]
            
            new_block = list(old_block)
            num_changes = rng.randint(1, min(3, k))
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val
            
            new_block.sort()
            
            if new_block == old_block:
                temp *= cooling
                continue
            
            new_bsi = compute_block_subs(new_block)
            
            valid = True
            if len(old_bsi) > 0:
                unique_mask = coverage[old_bsi] == 1
                if np.any(unique_mask):
                    unique_subs = old_bsi[unique_mask]
                    new_set = set(new_bsi.tolist()) if len(new_bsi) > 0 else set()
                    for us in unique_subs:
                        if us not in new_set:
                            valid = False
                            break
            
            if valid:
                if len(old_bsi) > 0:
                    np.add.at(coverage, old_bsi, -1)
                if len(new_bsi) > 0:
                    np.add.at(coverage, new_bsi, 1)
                current_blocks[idx] = new_block
                block_subs[idx] = new_bsi
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            
            temp *= cooling
            continue
        
        else:
            # Add a new block
            if len(current_blocks) < best_size + 5:
                best_block = None
                best_gain = -1
                
                for _ in range(10):
                    candidate = sorted(rng.choice(v, size=k, replace=False).tolist())
                    c_bsi = compute_block_subs(candidate)
                    if len(c_bsi) > 0:
                        gain = np.sum(coverage[c_bsi] == 0)
                    else:
                        gain = 0
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate
                
                if best_block is not None and best_gain > 0:
                    new_bsi = compute_block_subs(best_block)
                    if len(new_bsi) > 0:
                        np.add.at(coverage, new_bsi, 1)
                    current_blocks.append(best_block)
                    block_subs.append(new_bsi)
                    
                    new_size = len(current_blocks)
                    if new_size < best_size:
                        best_blocks = list(current_blocks)
                        best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
    
    return best_blocks


def _merge_blocks_pass(blocks, v, k, t, deadline):
    """
    Try to merge pairs of blocks: if the union of their t-subsets can be
    covered by a single k-block, replace both with that block.
    """
    if len(blocks) < 2:
        return blocks
    
    all_subs, sub_to_idx, n_subs = _build_sub_index(v, t)
    
    def compute_block_subs(block):
        return _block_to_sub_indices(block, sub_to_idx)
    
    block_subs = [compute_block_subs(b) for b in blocks]
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for bsi in block_subs:
        if len(bsi) > 0:
            np.add.at(coverage, bsi, 1)
    
    merged = True
    while merged and time.time() < deadline:
        merged = False
        new_blocks = list(blocks)
        new_block_subs = list(block_subs)
        
        i = 0
        while i < len(new_blocks) - 1:
            if time.time() > deadline:
                break
            
            b1_subs = new_block_subs[i]
            b2_subs = new_block_subs[i + 1]
            
            # Check if we can find a k-block that covers all t-subsets
            # currently uniquely covered by block i or block i+1
            unique_to_1 = np.zeros(n_subs, dtype=bool)
            unique_to_2 = np.zeros(n_subs, dtype=bool)
            if len(b1_subs) > 0:
                unique_to_1[b1_subs] = coverage[b1_subs] == 1
            if len(b2_subs) > 0:
                unique_to_2[b2_subs] = coverage[b2_subs] == 1
            
            # The union of unique subsets that must be in the new block
            must_cover = np.where(unique_to_1 | unique_to_2)[0]
            
            if len(must_cover) == 0:
                # Both blocks are fully redundant, just remove one
                if len(b1_subs) > 0:
                    np.add.at(coverage, b1_subs, -1)
                if len(b2_subs) > 0:
                    np.add.at(coverage, b2_subs, -1)
                new_blocks.pop(i)
                new_blocks.pop(i)
                new_block_subs.pop(i)
                new_block_subs.pop(i)
                merged = True
                continue
            
            # Collect all points that appear in must_cover subsets
            points_needed = set()
            for idx in must_cover:
                for p in all_subs[idx]:
                    points_needed.add(p)
            
            if len(points_needed) > k:
                i += 1
                continue
            
            # Try to find a k-block containing all needed points
            # Start with needed points, fill rest greedily
            base = sorted(points_needed)
            if len(base) > k:
                i += 1
                continue
            
            new_block = list(base)
            block_set = set(new_block)
            
            while len(new_block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx.get(candidate, -1)
                        if idx >= 0 and coverage[idx] <= 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                new_block.append(best_point)
                block_set.add(best_point)
            
            new_block.sort()
            new_bsi = compute_block_subs(new_block)
            
            # Verify: all must_cover subsets are in new_bsi
            new_set = set(new_bsi.tolist()) if len(new_bsi) > 0 else set()
            valid = all(m in new_set for m in must_cover)
            
            if valid:
                # Remove old blocks, add new one
                if len(b1_subs) > 0:
                    np.add.at(coverage, b1_subs, -1)
                if len(b2_subs) > 0:
                    np.add.at(coverage, b2_subs, -1)
                if len(new_bsi) > 0:
                    np.add.at(coverage, new_bsi, 1)
                
                new_blocks[i] = new_block
                new_blocks.pop(i + 1)
                new_block_subs[i] = new_bsi
                new_block_subs.pop(i + 1)
                merged = True
                continue
            
            i += 1
        
        blocks = new_blocks
        block_subs = new_block_subs
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for final validation
    
    # Step 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)
    
    # Step 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 3: Merge pass
    if time.time() < deadline - 30:
        blocks = _merge_blocks_pass(blocks, v, k, t, deadline - 25)
    
    # Step 4: Local search (first pass)
    if time.time() < deadline - 25:
        blocks = _local_search(blocks, v, k, t, deadline - 15, seed=42)
    
    # Step 5: Simulated annealing (second pass with different seed)
    if time.time() < deadline - 10:
        blocks = _sa_search(blocks, v, k, t, deadline - 5, seed=123)
    
    # Step 6: Final redundant removal
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 7: Final merge pass
    if time.time() < deadline - 2:
        blocks = _merge_blocks_pass(blocks, v, k, t, deadline)
    
    # Ensure validity: check all t-subsets are covered
    all_subs, sub_to_idx, n_subs = _build_sub_index(v, t)
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    # If not fully covered, add more blocks greedily
    while not (coverage > 0).all() and time.time() < deadline:
        uncovered_idx = np.where(coverage == 0)[0][0]
        sub = all_subs[uncovered_idx]
        block = list(sub)
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in sub_to_idx:
                        sidx = sub_to_idx[candidate]
                        if coverage[sidx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        for sub2 in itertools.combinations(block, t):
            sidx = sub_to_idx[sub2]
            coverage[sidx] += 1
    
    return blocks
# EVOLVE-BLOCK-END
