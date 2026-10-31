import time
import random
import itertools
import numpy as np
import sys

_CACHE = {}

def _get_orbit_id(subset, v):
    """
    Get the canonical representative of the orbit of a subset under Z_v.
    Returns the lexicographically smallest rotation.
    """
    s = sorted(subset)
    best = s
    for shift in range(1, v):
        rotated = [(x + shift) % v for x in s]
        rotated.sort()
        if rotated < best:
            best = rotated
    return tuple(best)

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    # 1. Process t-subsets (items to cover)
    t_subs = list(itertools.combinations(range(v), t))
    
    # Map each t-subset to its orbit ID
    orbit_reps = set()
    sub_to_orbit_rep = {}
    for sub in t_subs:
        rep = _get_orbit_id(sub, v)
        orbit_reps.add(rep)
        sub_to_orbit_rep[sub] = rep
        
    # Assign integer IDs to orbit reps
    orbit_reps_list = sorted(orbit_reps)
    rep_to_id = {rep: i for i, rep in enumerate(orbit_reps_list)}
    n_orbits_items = len(orbit_reps_list)
    
    # 2. Process k-subsets (blocks)
    k_subs = list(itertools.combinations(range(v), k))
    
    block_orbit_reps = set()
    for blk in k_subs:
        rep = _get_orbit_id(blk, v)
        block_orbit_reps.add(rep)
        
    block_orbits_list = sorted(block_orbit_reps)
    n_orbits_blocks = len(block_orbits_list)
    
    # For each block orbit, determine which item orbits it covers
    cov_masks = [0] * n_orbits_blocks
    
    for i, blk_rep in enumerate(block_orbits_list):
        mask = 0
        for t_sub in itertools.combinations(blk_rep, t):
            rep_t = sub_to_orbit_rep[t_sub]
            oid = rep_to_id[rep_t]
            mask |= (1 << oid)
        cov_masks[i] = mask
        
    _CACHE[key] = (block_orbits_list, cov_masks, n_orbits_blocks, n_orbits_items)
    return _CACHE[key]

def _solve_cover(cov_masks, n_orbits_items, n_orbits_blocks, deadline, rng):
    """
    Solve the set cover problem on the orbits.
    Returns a list of indices of selected block orbits.
    """
    # Greedy Phase
    uncovered = (1 << n_orbits_items) - 1
    selected = []
    selected_set = set()
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_score = -1
        
        for i in range(n_orbits_blocks):
            if i in selected_set:
                continue
            m = cov_masks[i]
            new_cov = m & uncovered
            if new_cov == 0:
                continue
            score = new_cov.bit_count()
            score += rng.random() * 0.01
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~cov_masks[best_idx]
        
    # Local Search Phase
    counts = [0] * n_orbits_items
    for idx in selected:
        m = cov_masks[idx]
        while m:
            lb = m & (-m)
            counts[lb.bit_length() - 1] += 1
            m &= m - 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try removing single blocks
        for pos in range(len(selected)):
            idx = selected[pos]
            m = cov_masks[idx]
            can_remove = True
            tmp = m
            while tmp:
                lb = tmp & (-tmp)
                if counts[lb.bit_length() - 1] == 1:
                    can_remove = False
                    break
                tmp &= tmp - 1
            if can_remove:
                tmp = m
                while tmp:
                    lb = tmp & (-tmp)
                    counts[lb.bit_length() - 1] -= 1
                    tmp &= tmp - 1
                selected.pop(pos)
                selected_set.remove(idx)
                improved = True
                break
                
        if improved:
            continue
            
        # 2. Try 1-for-1 swaps
        for pos in range(len(selected)):
            i = selected[pos]
            mi = cov_masks[i]
            
            # Find items that would become uncovered if we remove i
            critical_mask = 0
            tmp = mi
            while tmp:
                lb = tmp & (-tmp)
                oid = lb.bit_length() - 1
                if counts[oid] == 1:
                    critical_mask |= lb
                tmp &= tmp - 1
                
            if critical_mask == 0:
                continue
            
            # Find a block k not in selected that covers all critical items
            best_k = -1
            best_extra = -1
            for k in range(n_orbits_blocks):
                if k in selected_set:
                    continue
                mk = cov_masks[k]
                if (mk & critical_mask) == critical_mask:
                    extra = (mk & ~mi).bit_count()
                    if extra > best_extra:
                        best_extra = extra
                        best_k = k
                        
            if best_k != -1:
                # Perform swap: remove i, add k
                tmp = mi
                while tmp:
                    lb = tmp & (-tmp)
                    counts[lb.bit_length() - 1] -= 1
                    tmp &= tmp - 1
                    
                mk = cov_masks[best_k]
                tmp = mk
                while tmp:
                    lb = tmp & (-tmp)
                    counts[lb.bit_length() - 1] += 1
                    tmp &= tmp - 1
                    
                selected[pos] = best_k
                selected_set.remove(i)
                selected_set.add(best_k)
                improved = True
                break
                
        if improved:
            continue
            
        # 3. Try 2-for-1 swaps
        sel_list = list(selected)
        n_sel = len(sel_list)
        
        for p1 in range(n_sel):
            for p2 in range(p1 + 1, n_sel):
                i = sel_list[p1]
                j = sel_list[p2]
                mi = cov_masks[i]
                mj = cov_masks[j]
                
                # Calculate critical mask
                crit_mask = 0
                combined = mi | mj
                tmp = combined
                while tmp:
                    lb = tmp & (-tmp)
                    oid = lb.bit_length() - 1
                    c = counts[oid]
                    if c == 1:
                        if (mi & lb) or (mj & lb):
                            crit_mask |= lb
                    elif c == 2:
                        if (mi & lb) and (mj & lb):
                            crit_mask |= lb
                    tmp &= tmp - 1
                    
                if crit_mask == 0:
                    break
                
                # Find a block k not in selected that covers crit_mask
                best_k = -1
                for k in range(n_orbits_blocks):
                    if k in selected_set:
                        continue
                    mk = cov_masks[k]
                    if (mk & crit_mask) == crit_mask:
                        best_k = k
                        break
                        
                if best_k != -1:
                    # Perform swap
                    tmp = mi
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] -= 1
                        tmp &= tmp - 1
                    tmp = mj
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] -= 1
                        tmp &= tmp - 1
                        
                    mk = cov_masks[best_k]
                    tmp = mk
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] += 1
                        tmp &= tmp - 1
                        
                    selected.pop(p2)
                    selected.pop(p1)
                    selected_set.remove(i)
                    selected_set.remove(j)
                    
                    selected.append(best_k)
                    selected_set.add(best_k)
                    
                    improved = True
                    break
                    
            if improved:
                break
                
        if improved:
            continue
            
        # 4. Try 2-for-2 swaps
        for p1 in range(n_sel):
            for p2 in range(p1 + 1, n_sel):
                i = sel_list[p1]
                j = sel_list[p2]
                mi = cov_masks[i]
                mj = cov_masks[j]
                
                # Calculate items that would become uncovered if we remove both i and j
                crit_mask = 0
                combined = mi | mj
                tmp = combined
                while tmp:
                    lb = tmp & (-tmp)
                    oid = lb.bit_length() - 1
                    c = counts[oid]
                    if c == 1:
                        crit_mask |= lb
                    elif c == 2:
                        if (mi & lb) and (mj & lb):
                            crit_mask |= lb
                    tmp &= tmp - 1
                    
                if crit_mask == 0:
                    break
                
                # Find two blocks k, l not in selected that together cover crit_mask
                best_pair = None
                for k in range(n_orbits_blocks):
                    if k in selected_set:
                        continue
                    mk = cov_masks[k]
                    remaining = crit_mask & ~mk
                    if remaining == 0:
                        # k alone covers everything, but we want 2-for-2
                        # This would be a 2-for-1, which we already tried
                        continue
                    for l in range(n_orbits_blocks):
                        if l in selected_set or l == k:
                            continue
                        ml = cov_masks[l]
                        if (ml & remaining) == remaining:
                            # Found a valid pair
                            extra_k = (mk & ~(mi | mj)).bit_count()
                            extra_l = (ml & ~(mi | mj)).bit_count()
                            best_pair = (k, l, extra_k + extra_l)
                            break
                    if best_pair:
                        break
                        
                if best_pair:
                    k, l, _ = best_pair
                    mk = cov_masks[k]
                    ml = cov_masks[l]
                    
                    # Perform swap
                    tmp = mi
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] -= 1
                        tmp &= tmp - 1
                    tmp = mj
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] -= 1
                        tmp &= tmp - 1
                        
                    tmp = mk
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] += 1
                        tmp &= tmp - 1
                    tmp = ml
                    while tmp:
                        lb = tmp & (-tmp)
                        counts[lb.bit_length() - 1] += 1
                        tmp &= tmp - 1
                        
                    selected.pop(p2)
                    selected.pop(p1)
                    selected_set.remove(i)
                    selected_set.remove(j)
                    
                    selected.append(k)
                    selected.append(l)
                    selected_set.add(k)
                    selected_set.add(l)
                    
                    improved = True
                    break
                    
            if improved:
                break
                
    return selected

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    rng = random.Random(12345)
    
    block_orbits, cov_masks, n_orbits_blocks, n_orbits_items = _prepare(v, k, t)
    
    # Solve the orbit covering problem
    selected_orbits = _solve_cover(cov_masks, n_orbits_items, n_orbits_blocks, deadline, rng)
    
    # Expand orbits to actual blocks
    blocks = []
    for idx in selected_orbits:
        rep = block_orbits[idx]
        for shift in range(v):
            blk = tuple((x + shift) % v for x in rep)
            blocks.append(sorted(blk))
            
    return blocks
