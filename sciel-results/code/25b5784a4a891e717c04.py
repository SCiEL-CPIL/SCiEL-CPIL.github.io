import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    # Deadline guard: 150s limit, use 145s to be safe
    deadline = time.time() + 145
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all possible blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    n_blocks = len(all_blocks)
    
    # Precompute coverage mask for each possible block
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Map from t-subset index to list of block indices that cover it
    # This is useful for finding candidates to cover a specific t-subset
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for i in range(n_blocks):
        mask = block_masks[i]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_to_blocks[idx].append(i)
            m ^= lsb

    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    def _remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
            
        cover_count = [0] * n_tsubs
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        
        for bi in block_indices:
            mask = block_masks[bi]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    mask = block_masks[bi]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break # Restart loop
                    
        return blocks

    def _local_search(blocks, deadline):
        if not blocks:
            return blocks
            
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        improved = True
        max_iterations = 200
        iter_count = 0
        
        while improved and time.time() < deadline - 10 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Shuffle block order to randomize search
            order = list(range(len(blocks)))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 10:
                    break
                if i >= len(blocks):
                    continue

                bi = block_indices[i]
                mask = block_masks[bi]
                
                # Tentatively remove block i
                uncovered = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        uncovered.append(idx)
                    m ^= lsb
                
                if not uncovered:
                    # Redundant block, remove it
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    blocks.pop(i)
                    block_indices.pop(i)
                    improved = True
                    break # Restart iteration with new list sizes
                
                # Try to cover 'uncovered' t-subsets with a single new block
                # Find a block that covers the most 'uncovered' t-subsets
                best_block_idx = -1
                best_gain = 0
                # Candidates must not be in the current set (indices 0..len(blocks)-1 excluding i)
                # We can just iterate all blocks and check if they are in the current set?
                # Or maintain a set of current block indices for O(1) lookup.
                current_indices_set = set(block_indices)
                
                # To speed up, we can check only blocks that cover at least one uncovered t-subset
                # Union of tsub_to_blocks for all uncovered
                candidates = set()
                for idx in uncovered:
                    candidates.update(tsub_to_blocks[idx])
                
                # Filter out current blocks
                for cand in candidates:
                    if cand in current_indices_set:
                        continue
                    
                    # Calculate gain: how many uncovered t-subsets does this candidate cover?
                    cand_mask = block_masks[cand]
                    gain = 0
                    m = cand_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1: # It was covered only by block i, so now uncovered
                            gain += 1
                        m ^= lsb
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = cand
                        
                if best_block_idx != -1 and best_gain == len(uncovered):
                    # We can cover all uncovered t-subsets with one new block
                    # Perform the swap
                    # 1. Remove block i
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    
                    # 2. Add new block
                    new_mask = block_masks[best_block_idx]
                    m = new_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                        
                    blocks[i] = all_blocks[best_block_idx]
                    block_indices[i] = best_block_idx
                    improved = True
                    break # Restart iteration
        
        return blocks

    # Main Loop: Randomized Greedy with Bitmasks
    for restart in range(100):
        if time.time() > deadline - 20:
            break

        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        
        # Randomize the order of t-subsets to break symmetry
        order = list(range(n_tsubs))
        rng.shuffle(order)
        
        while uncovered_mask:
            if time.time() > deadline - 10:
                break
            
            # Find the first uncovered t-subset in the randomized order
            seed_idx = -1
            for idx in order:
                if (uncovered_mask >> idx) & 1:
                    seed_idx = idx
                    break
                    
            if seed_idx == -1:
                break
                
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_point = -1
                tie_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: how many new t-subsets does adding x cover?
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > -1:
                        tie_points.append(x)
                        
                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                        block.append(best_point)
                        block_set.add(best_point)
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            bi = block_to_idx[tuple(block)]
            block_mask = block_masks[bi]
            uncovered_mask &= ~block_mask
            
        if time.time() > deadline - 10 or uncovered_mask:
            continue
            
        # Post-processing
        blocks = _remove_redundant(blocks, deadline)
        
        if time.time() < deadline - 15:
            blocks = _local_search(blocks, deadline)
            blocks = _remove_redundant(blocks, deadline)
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback: Simple greedy
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        while uncovered_mask:
            seed_idx = 0
            while (uncovered_mask >> seed_idx) & 1 == 0:
                seed_idx += 1
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
            block.sort()
            blocks.append(block)
            bi = block_to_idx[tuple(block)]
            uncovered_mask &= ~block_masks[bi]
        best_blocks = blocks

    # Final cleanup
    best_blocks = _remove_redundant(best_blocks, deadline)
    
    return best_blocks
