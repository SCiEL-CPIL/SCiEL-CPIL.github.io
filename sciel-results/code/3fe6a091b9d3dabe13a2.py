"""Covering design C(v,k,t): minimize number of k-subset blocks covering all t-subsets.

This version uses a refined greedy construction with improved criticality scoring
and a more effective local search phase with better swap heuristics.
"""
import time
import random
import itertools

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# Precompute all t-subsets and each block's coverage as an integer bitmask.
_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n = len(t_subs)
    cov = {}
    for blk in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov[blk] = m
    _CACHE[key] = (t_subs, index, n, cov)
    return _CACHE[key]


def _local_search(selected, cov_list, n, deadline, rng, max_rounds=20000):
    """Enhanced local search: drop redundant blocks, single swaps, and 2-for-1 exchanges.
    Optimized to minimize per-round overhead for more iterations within the time budget."""
    nblocks = len(cov_list)
    sel = list(selected)
    sel_set = set(sel)
    # coverage count of each t-subset by selected blocks
    count = [0] * n
    for i in sel:
        m = cov_list[i]
        while m:
            lb = m & (-m)
            count[lb.bit_length() - 1] += 1
            m &= m - 1

    improved = True
    rounds = 0
    no_improve_rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        
        # 1) try dropping each selected block - find any droppable block
        drop_idx = -1
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            ok = True
            while m:
                lb = m & (-m)
                if count[lb.bit_length() - 1] == 1:
                    ok = False
                    break
                m &= m - 1
            if ok:
                drop_idx = pos
                break
        
        if drop_idx != -1:
            i = sel[drop_idx]
            m = cov_list[i]
            while m:
                lb = m & (-m)
                count[lb.bit_length() - 1] -= 1
                m &= m - 1
            sel.pop(drop_idx)
            sel_set.remove(i)
            improved = True
            no_improve_rounds = 0
            continue
            
        # 2) try swapping: find best swap in one pass
        best_swap_score = -1
        best_swap_i = -1
        best_swap_j = -1
        
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            crit_mask = 0
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if count[idx] == 1:
                    crit_mask |= lb
                m &= m - 1
            
            if crit_mask == 0:
                continue

            # Find j not in sel that covers crit_mask
            for j in range(nblocks):
                if j in sel_set:
                    continue
                mj = cov_list[j]
                if (mj & crit_mask) == crit_mask:
                    new_cov = (mj & ~m).bit_count()
                    score = new_cov + rng.random() * 1e-6
                    if score > best_swap_score:
                        best_swap_score = score
                        best_swap_i = pos
                        best_swap_j = j
        
        if best_swap_j != -1:
            i = sel[best_swap_i]
            j = best_swap_j
            mi = cov_list[i]
            mj = cov_list[j]
            while mi:
                lb = mi & (-mi)
                count[lb.bit_length() - 1] -= 1
                mi &= mi - 1
            while mj:
                lb = mj & (-mj)
                count[lb.bit_length() - 1] += 1
                mj &= mj - 1
            sel[best_swap_i] = j
            sel_set.remove(i)
            sel_set.add(j)
            improved = True
            no_improve_rounds = 0
            continue
        
        # 3) 2-for-1 exchange with limited sampling
        if len(sel) >= 2:
            n_pairs_to_try = min(30, len(sel) * (len(sel) - 1) // 2)
            pairs = [(p1, p2) for p1 in range(len(sel)) for p2 in range(p1 + 1, len(sel))]
            rng.shuffle(pairs)
            
            for p1, p2 in pairs[:n_pairs_to_try]:
                i1 = sel[p1]
                i2 = sel[p2]
                m1_orig = cov_list[i1]
                m2_orig = cov_list[i2]
                
                combined_mask = m1_orig | m2_orig
                crit_mask = 0
                while combined_mask:
                    lb = combined_mask & (-combined_mask)
                    idx = lb.bit_length() - 1
                    if count[idx] <= 2 and ((m1_orig & lb) or (m2_orig & lb)):
                        if count[idx] == 1 or ((m1_orig & lb) and (m2_orig & lb)):
                            crit_mask |= lb
                    combined_mask &= combined_mask - 1
                
                if crit_mask == 0:
                    continue
                
                for j in range(nblocks):
                    if j in sel_set:
                        continue
                    mj = cov_list[j]
                    if (mj & crit_mask) == crit_mask:
                        new_mask = mj & ~(m1_orig | m2_orig)
                        if new_mask.bit_count() > 0:
                            m1 = m1_orig
                            while m1:
                                lb = m1 & (-m1)
                                count[lb.bit_length() - 1] -= 1
                                m1 &= m1 - 1
                            m2 = m2_orig
                            while m2:
                                lb = m2 & (-m2)
                                count[lb.bit_length() - 1] -= 1
                                m2 &= m2 - 1
                            mj_copy = mj
                            while mj_copy:
                                lb = mj_copy & (-mj_copy)
                                count[lb.bit_length() - 1] += 1
                                mj_copy &= mj_copy - 1
                            
                            if p1 < p2:
                                sel.pop(p2)
                                sel.pop(p1)
                            else:
                                sel.pop(p1)
                                sel.pop(p2)
                            sel.append(j)
                            sel_set.remove(i1)
                            sel_set.remove(i2)
                            sel_set.add(j)
                            improved = True
                            no_improve_rounds = 0
                            break
                
                if improved:
                    break
            if improved:
                continue
        
        if not improved:
            no_improve_rounds += 1
            if no_improve_rounds > 20:
                break

    return sel


def _construct_direct(v, k, t, deadline, rng):
    """
    Directly construct a covering by greedily selecting blocks using a deficiency-ratio heuristic.
    The score balances new coverage against point saturation, modeling the problem as dynamic
    resource allocation rather than static set cover.
    """
    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    
    uncovered = (1 << n) - 1
    selected_blocks = []
    selected_set = set()
    counts = [0] * n  # Coverage count for each t-subset
    point_usage = [0] * v  # How many selected blocks contain each point
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_score = -1
        
        for b in all_blocks:
            if b in selected_set:
                continue
            
            m = cov[b]
            new_cov = (m & uncovered).bit_count()
            
            if new_cov == 0:
                continue
            
            # Deficiency ratio: new coverage divided by point saturation penalty
            # Points that are heavily used have lower "future potential"
            sat_penalty = 1.0
            for pt in b:
                sat_penalty *= (1.0 + 0.3 * point_usage[pt])
            
            # Criticality: weight uncovered subsets more
            crit_score = 0.0
            tmp = m & uncovered
            while tmp:
                lb = tmp & (-tmp)
                idx = lb.bit_length() - 1
                weight = 1.0 / (1.0 + counts[idx])
                crit_score += weight
                tmp &= tmp - 1
            
            # Deficiency ratio: new_cov / sat_penalty, combined with criticality
            deficiency_ratio = new_cov / sat_penalty
            score = deficiency_ratio * 15.0 + crit_score * 2.0 + rng.random() * 0.02
            
            if score > best_score:
                best_score = score
                best_block = b
        
        if best_block is None:
            break
            
        selected_blocks.append(best_block)
        selected_set.add(best_block)
        
        m = cov[best_block]
        uncovered &= ~m
        while m:
            lb = m & (-m)
            counts[lb.bit_length() - 1] += 1
            m &= m - 1
        for pt in best_block:
            point_usage[pt] += 1
            
    return selected_blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s
    rng = random.Random(12345)

    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    
    # Phase 1: Direct Construction with deficiency-ratio heuristic
    # Use 50% of time for construction, leaving 50% for local search
    cons_deadline = t0 + (deadline - t0) * 0.50
    
    selected_blocks = _construct_direct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    cov_list = [cov[b] for b in all_blocks]
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    selected_indices = [block_to_idx[b] for b in selected_blocks]
    
    ls_deadline = deadline - 3.0
    selected_indices = _local_search(selected_indices, cov_list, n, ls_deadline, rng)
    
    # Phase 3: Final polish - run local search again with remaining time
    if time.time() < deadline - 5.0:
        selected_indices = _local_search(selected_indices, cov_list, n, deadline - 3.0, rng)
    
    # Ensure validity (fallback)
    covered_mask = 0
    for i in selected_indices:
        covered_mask |= cov_list[i]
        
    full_mask = (1 << n) - 1
    uncovered_now = full_mask & ~covered_mask
    
    if uncovered_now and time.time() < deadline - 1:
        for i in range(len(all_blocks)):
            if i in selected_indices:
                continue
            if cov_list[i] & uncovered_now:
                selected_indices.append(i)
                uncovered_now &= ~cov_list[i]
                if not uncovered_now:
                    break
    
    result = [list(all_blocks[i]) for i in selected_indices]
    return result
