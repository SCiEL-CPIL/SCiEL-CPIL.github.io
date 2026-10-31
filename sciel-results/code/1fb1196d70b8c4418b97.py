"""Covering design C(v,k,t): minimize number of k-subset blocks covering all t-subsets.

Structural redesign: Direct greedy construction with criticality bias.
We build the covering set block-by-block, prioritizing the coverage of
currently uncovered or critically covered t-subsets. This avoids the
inefficiency of a separate generation/selection phase and directly targets
bottlenecks.
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
_T_SUBSETS = []
_T_INDEX = {}
_BLOCK_COV = {}
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


def _local_search(selected, cov_list, n, deadline, rng, max_rounds=8000):
    """Enhanced local search: drop redundant blocks, single swaps, and 2-for-1 exchanges."""
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
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        # 1) try dropping each selected block
        drop_idx = -1
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            # check no t-subset becomes uncovered
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
            continue
            
        # 2) try swapping: find a block j not in sel that covers the critical subsets of some i in sel
        # and allows dropping i (or another block). 
        swap_done = False
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

            # Find j not in sel that covers crit_mask and maximizes additional coverage
            for j in range(nblocks):
                if j in sel_set:
                    continue
                mj = cov_list[j]
                if (mj & crit_mask) == crit_mask:
                    # Calculate score: how many new subsets j covers beyond what i covers
                    # plus how many subsets become more redundantly covered
                    new_cov = (mj & ~m).bit_count()
                    # Prefer swaps that add more unique coverage
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
            # Remove i
            while mi:
                lb = mi & (-mi)
                count[lb.bit_length() - 1] -= 1
                mi &= mi - 1
            # Add j
            while mj:
                lb = mj & (-mj)
                count[lb.bit_length() - 1] += 1
                mj &= mj - 1
            sel[best_swap_i] = j
            sel_set.remove(i)
            sel_set.add(j)
            improved = True
            continue
        
        # 3) 2-for-1 exchange: try dropping 2 blocks and adding 1 block that covers their union's critical parts
        # This is more expensive but can find better solutions
        if len(sel) >= 2:
            # Sample a few pairs to try
            n_pairs_to_try = min(20, len(sel) * (len(sel) - 1) // 2)
            pairs = []
            for p1 in range(len(sel)):
                for p2 in range(p1 + 1, len(sel)):
                    pairs.append((p1, p2))
            rng.shuffle(pairs)
            
            for p1, p2 in pairs[:n_pairs_to_try]:
                i1 = sel[p1]
                i2 = sel[p2]
                m1 = cov_list[i1]
                m2 = cov_list[i2]
                
                # Find critical subsets that would be uncovered if we remove both i1 and i2
                # A subset is critical if it's covered exactly once by i1 or i2 (and not by others)
                combined_mask = m1 | m2
                crit_mask = 0
                while combined_mask:
                    lb = combined_mask & (-combined_mask)
                    idx = lb.bit_length() - 1
                    # Check if this subset is covered only by i1 or i2
                    if count[idx] <= 2:
                        # Need to verify it's actually covered by i1 or i2
                        if (m1 & lb) or (m2 & lb):
                            if count[idx] == 1:
                                crit_mask |= lb
                            elif count[idx] == 2:
                                # Covered by both i1 and i2, or one of them and another
                                # If covered by i1 and i2, removing both uncovers it
                                if (m1 & lb) and (m2 & lb):
                                    crit_mask |= lb
                    combined_mask &= combined_mask - 1
                
                if crit_mask == 0:
                    continue
                
                # Find j not in sel that covers crit_mask
                for j in range(nblocks):
                    if j in sel_set:
                        continue
                    mj = cov_list[j]
                    if (mj & crit_mask) == crit_mask:
                        # Check if this swap is beneficial: j should cover more than the union of i1 and i2
                        # Remove i1 and i2, add j
                        # First simulate: what's the net change?
                        new_mask = mj & ~(m1 | m2)
                        if new_mask.bit_count() > 0:
                            # Perform the swap
                            # Remove i1
                            while m1:
                                lb = m1 & (-m1)
                                count[lb.bit_length() - 1] -= 1
                                m1 &= m1 - 1
                            # Remove i2
                            m2_copy = cov_list[i2]
                            while m2_copy:
                                lb = m2_copy & (-m2_copy)
                                count[lb.bit_length() - 1] -= 1
                                m2_copy &= m2_copy - 1
                            # Add j
                            mj_copy = cov_list[j]
                            while mj_copy:
                                lb = mj_copy & (-mj_copy)
                                count[lb.bit_length() - 1] += 1
                                mj_copy &= mj_copy - 1
                            
                            # Update sel
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
                            break
                
                if improved:
                    break
            if improved:
                continue

    return sel


def _construct_direct(v, k, t, deadline, rng):
    """
    Directly construct a covering by greedily selecting blocks.
    Prioritizes covering uncovered subsets with criticality bias.
    """
    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    
    uncovered = (1 << n) - 1
    selected_blocks = []
    selected_set = set()
    counts = [0] * n  # Coverage count for each t-subset
    
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
            
            # Criticality bias: reward covering subsets that are currently uncovered
            # or have low coverage count
            # Compute a weighted score based on how critical each newly covered subset is
            crit_score = 0
            tmp = m & uncovered
            while tmp:
                lb = tmp & (-tmp)
                idx = lb.bit_length() - 1
                # Higher weight for subsets with lower current coverage
                weight = 1.0 / (1.0 + counts[idx])
                crit_score += weight
                tmp &= tmp - 1
            
            # Combine: primary is new coverage count, secondary is criticality
            score = new_cov * 10.0 + crit_score + rng.random() * 0.1
            
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
            
    return selected_blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    
    # Phase 1: Direct Construction
    # Use 70% of time for construction, leaving 30% for local search
    cons_deadline = t0 + (deadline - t0) * 0.7
    
    selected_blocks = _construct_direct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    cov_list = [cov[b] for b in all_blocks]
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    selected_indices = [block_to_idx[b] for b in selected_blocks]
    
    ls_deadline = deadline - 2.0
    selected_indices = _local_search(selected_indices, cov_list, n, ls_deadline, rng)
    
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
