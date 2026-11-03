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
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage mask and t-subset index list for each possible block
    block_masks = [0] * len(all_blocks)
    block_tsubs = [[] for _ in range(len(all_blocks))]
    for i, b in enumerate(all_blocks):
        mask = 0
        tsub_list = []
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
            tsub_list.append(idx)
        block_masks[i] = mask
        block_tsubs[i] = tsub_list
    
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    def _remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
            
        n_tsubs_local = n_tsubs
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs_local
        
        for bi in block_indices:
            for idx in block_tsubs[bi]:
                cover_count[idx] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                redundant = True
                for idx in block_tsubs[bi]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    
                if redundant:
                    for idx in block_tsubs[bi]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break
                    
        return blocks

    def _local_search(blocks, deadline):
        if not blocks:
            return blocks
            
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            for idx in block_tsubs[bi]:
                cover_count[idx] += 1
                
        improved = True
        max_iterations = 80
        iter_count = 0
        
        while improved and time.time() < deadline - 10 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Try to remove a redundant block
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                
                bi = block_indices[i]
                criticals = []
                for idx in block_tsubs[bi]:
                    if cover_count[idx] == 1:
                        criticals.append(idx)
                
                if not criticals:
                    # Block i is redundant
                    for idx in block_tsubs[bi]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 1-point swaps to cover critical t-subsets
            found_swap = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                if found_swap:
                    break
                    
                bi = block_indices[i]
                criticals = []
                for idx in block_tsubs[bi]:
                    if cover_count[idx] == 1:
                        criticals.append(idx)
                
                if not criticals:
                    continue
                    
                # For each critical t-subset, try to find a block to swap
                for crit_idx in criticals:
                    if found_swap:
                        break
                    crit_tsub = all_tsubs[crit_idx]
                    crit_set = set(crit_tsub)
                    
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if time.time() > deadline - 10:
                            break
                        if found_swap:
                            break
                            
                        bj = blocks[j]
                        bj_set = set(bj)
                        common = bj_set.intersection(crit_set)
                        
                        if len(common) >= t - 1:
                            missing = [p for p in crit_tsub if p not in bj_set]
                            if len(missing) == 1:
                                extra = [p for p in bj if p not in common]
                                for rem_p in extra:
                                    if found_swap:
                                        break
                                    new_bj = sorted([p if p != rem_p else missing[0] for p in bj])
                                    if new_bj == sorted(bj):
                                        continue
                                    new_bi = block_to_idx.get(tuple(new_bj))
                                    if new_bi is None:
                                        continue
                                    
                                    old_tsubs = set(block_tsubs[block_indices[j]])
                                    new_tsubs = set(block_tsubs[new_bi])
                                    
                                    lost = old_tsubs - new_tsubs
                                    gained = new_tsubs - old_tsubs
                                    
                                    valid = True
                                    for idx in lost:
                                        if cover_count[idx] == 1:
                                            valid = False
                                            break
                                    
                                    if valid:
                                        for idx in lost:
                                            cover_count[idx] -= 1
                                        for idx in gained:
                                            cover_count[idx] += 1
                                        
                                        blocks[j] = new_bj
                                        block_indices[j] = new_bi
                                        improved = True
                                        found_swap = True
                                        break
                            
                            # Try 2-point swaps
                            if not found_swap and len(common) >= t - 2:
                                missing = [p for p in crit_tsub if p not in bj_set]
                                if len(missing) == 2:
                                    extra = [p for p in bj if p not in common]
                                    if len(extra) >= 2:
                                        for rem1, rem2 in itertools.combinations(extra, 2):
                                            if found_swap:
                                                break
                                            new_bj = sorted([p for p in bj if p != rem1 and p != rem2] + missing)
                                            if len(set(new_bj)) != k:
                                                continue
                                            new_bi = block_to_idx.get(tuple(new_bj))
                                            if new_bi is None:
                                                continue
                                            
                                            old_tsubs = set(block_tsubs[block_indices[j]])
                                            new_tsubs = set(block_tsubs[new_bi])
                                            
                                            lost = old_tsubs - new_tsubs
                                            gained = new_tsubs - old_tsubs
                                            
                                            valid = True
                                            for idx in lost:
                                                if cover_count[idx] == 1:
                                                    valid = False
                                                    break
                                            
                                            if valid:
                                                for idx in lost:
                                                    cover_count[idx] -= 1
                                                for idx in gained:
                                                    cover_count[idx] += 1
                                                
                                                blocks[j] = new_bj
                                                block_indices[j] = new_bi
                                                improved = True
                                                found_swap = True
                                                break
                if found_swap:
                    break

        return blocks

    # Precompute for each point x, the list of t-subsets containing x
    # This speeds up greedy gain computation
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(idx)
    
    # Main Loop: Randomized Greedy with Bitmasks
    max_restarts = 200
    for restart in range(max_restarts):
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
                    
                    # Calculate gain using precomputed point_tsubs
                    gain = 0
                    # A t-subset is covered if it contains x and t-1 points from current block
                    # Iterate over t-subsets containing x
                    for idx in point_tsubs[x]:
                        ts = all_tsubs[idx]
                        # Check if ts contains x and t-1 points from block
                        # Since ts contains x by construction, check if t-1 other points are in block
                        count_in_block = 0
                        for p in ts:
                            if p != x and p in block_set:
                                count_in_block += 1
                                if count_in_block == t - 1:
                                    break
                        if count_in_block == t - 1:
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
                    for idx in point_tsubs[x]:
                        ts = all_tsubs[idx]
                        count_in_block = 0
                        for p in ts:
                            if p != x and p in block_set:
                                count_in_block += 1
                                if count_in_block == t - 1:
                                    break
                        if count_in_block == t - 1:
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
