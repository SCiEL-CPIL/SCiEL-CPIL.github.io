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


def _local_search(selected, cov_list, n, deadline, rng, max_rounds=4000):
    """Block-exchange local search: drop a redundant block, or swap one in for one out."""
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
        # Simplified: For each i in sel, find critical subsets. If there exists j not in sel 
        # that covers all critical subsets of i, swap i->j. Then retry dropping.
        # To save time, we only do this if we haven't found a drop.
        swap_done = False
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
                    # Swap i with j
                    # Remove i
                    mi = cov_list[i]
                    while mi:
                        lb = mi & (-mi)
                        count[lb.bit_length() - 1] -= 1
                        mi &= mi - 1
                    # Add j
                    while mj:
                        lb = mj & (-mj)
                        count[lb.bit_length() - 1] += 1
                        mj &= mj - 1
                    sel[pos] = j
                    sel_set.remove(i)
                    sel_set.add(j)
                    improved = True
                    swap_done = True
                    break
            if swap_done:
                break
    return sel


def _construct_direct(v, k, t, deadline, rng):
    """
    Directly construct a covering by greedily selecting blocks.
    Prioritizes covering uncovered subsets.
    """
    t_subs, index, n, cov = _prepare(v, k, t)
    all_blocks = list(cov.keys())
    # Pre-compute bit counts for speed if possible, or just use .bit_count()
    
    uncovered = (1 << n) - 1
    selected_blocks = []
    selected_set = set()
    counts = [0] * n  # Coverage count for each t-subset
    
    # To speed up selection, we can maintain a list of "active" blocks that cover at least one uncovered subset.
    # However, checking all blocks is C(19,7) = 50388. 
    # 50388 * (bit operations) is fast enough for a few hundred iterations.
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_score = -1
        
        # Score: number of uncovered subsets covered + small penalty for redundant coverage
        # We want to maximize coverage of uncovered sets.
        
        for b in all_blocks:
            if b in selected_set:
                continue
            
            m = cov[b]
            # Count how many currently uncovered subsets this block covers
            # (m & uncovered).bit_count()
            new_cov = (m & uncovered).bit_count()
            
            if new_cov == 0:
                continue
            
            # Tie-breaking: prefer blocks that cover fewer already-covered subsets?
            # Or just random tie-break for diversity in early stages?
            # Let's stick to max new coverage.
            
            # Add small random noise to break ties and explore
            score = new_cov + rng.random() * 1e-6
            
            if score > best_score:
                best_score = score
                best_block = b
        
        if best_block is None:
            # Should not happen if uncovered is not zero and we have all blocks
            break
            
        selected_blocks.append(best_block)
        selected_set.add(best_block)
        
        # Update uncovered and counts
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
    # Use 80% of time for construction, leaving 20% for local search
    cons_deadline = t0 + (deadline - t0) * 0.8
    
    # We might run multiple random restarts if time permits?
    # For now, single run with random tie-breaking.
    
    selected_blocks = _construct_direct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    # Convert to indices or keep as tuples? 
    # _local_search expects indices into cov_list.
    # Let's create a cov_list from all_blocks to keep indices consistent.
    cov_list = [cov[b] for b in all_blocks]
    # Map block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    selected_indices = [block_to_idx[b] for b in selected_blocks]
    
    ls_deadline = deadline - 2.0
    selected_indices = _local_search(selected_indices, cov_list, n, ls_deadline, rng)
    
    # Ensure validity (fallback)
    # Check if any uncovered remain
    covered_mask = 0
    for i in selected_indices:
        covered_mask |= cov_list[i]
        
    full_mask = (1 << n) - 1
    uncovered_now = full_mask & ~covered_mask
    
    if uncovered_now and time.time() < deadline - 1:
        # Greedily add any block that helps
        # This is a safety net.
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
