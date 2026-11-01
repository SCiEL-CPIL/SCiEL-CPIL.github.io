"""Constraint-programming (backtracking + branch-and-bound) for covering C(24,7,3)."""
import itertools
import random
import time
import sys

# EVOLVE-BLOCK-START
import numpy as np

# Precompute: for each point, the list of 3-subsets (as bitmasks) containing it.
# v=24, t=3. 24C3 = 2024 t-sets.
_V = 24
_T = 3
_K = 7
_ALL_TSETS = list(itertools.combinations(range(_V), _T))
_N_TSETS = len(_ALL_TSETS)

# Map each t-set to a bit index
_TS_TO_IDX = {ts: i for i, ts in enumerate(_ALL_TSETS)}

# For each point x, the list of t-set indices that contain x
_PT_TO_TSETS = [[] for _ in range(_V)]
for i, ts in enumerate(_ALL_TSETS):
    for x in ts:
        _PT_TO_TSETS[x].append(i)

# For each (t-1)-subset (pair), the list of t-set indices containing that pair
# Used for fast gain computation: a block covers t-sets; the "new" t-sets a new
# point x adds to a partial block B are those t-sets containing x and 2 points of B
# that are still uncovered.
_PAIR_TO_TSETS = {}
for i, ts in enumerate(_ALL_TSETS):
    for a, b in itertools.combinations(ts, _T - 1):
        key = (a, b)
        _PAIR_TO_TSETS.setdefault(key, []).append(i)

# Precompute for each pair (a,b) the bitmask of t-sets containing it (as a Python int bitmask)
# 2024 bits fits in a Python int.
_PAIR_MASK = {}
for key, idxs in _PAIR_TO_TSETS.items():
    m = 0
    for i in idxs:
        m |= (1 << i)
    _PAIR_MASK[key] = m

# Precompute for each point x the list of pairs (x, y) for y != x, and their masks
# Actually we'll compute gain on the fly.


def _greedy_cover(v, k, t, rng, time_limit):
    """Fast greedy covering to get an initial upper bound."""
    deadline = time.time() + time_limit
    uncovered = (1 << _N_TSETS) - 1  # all t-sets uncovered
    blocks = []
    while uncovered and time.time() < deadline:
        # Find an uncovered t-set
        # Get lowest set bit
        low = uncovered & (-uncovered)
        idx = low.bit_length() - 1
        ts = _ALL_TSETS[idx]
        block = list(ts)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # gain = number of uncovered t-sets containing x and (t-1) pts of block
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x in S:
                        continue
                    key = tuple(sorted(S + (x,)))
                    m = _PAIR_MASK.get((S[0], S[1]), 0) if t - 1 == 2 else 0
                    # Actually for t=3, S is a pair. The t-set is S + (x,).
                    # We need the specific t-set index.
                    # Let's just compute the t-set index directly.
                    tset = tuple(sorted(S + (x,)))
                    tidx = _TS_TO_IDX[tset]
                    if (uncovered >> tidx) & 1:
                        gain += 1
                gain += rng.random() * 0.01
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        # Update uncovered
        for ts2 in itertools.combinations(block, t):
            tidx = _TS_TO_IDX[tuple(sorted(ts2))]
            uncovered &= ~(1 << tidx)
    return blocks


def _is_valid(blocks, v, k, t):
    covered = 0
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, t):
            tidx = _TS_TO_IDX[tuple(sorted(ts))]
            covered |= (1 << tidx)
    return covered == ((1 << _N_TSETS) - 1)


def _backtrack_cover(v, k, t, upper_bound, time_limit, rng):
    """
    Backtracking search with branch-and-bound.
    State: uncovered bitmask, current list of blocks, depth.
    We try to find a covering with fewer than upper_bound blocks.
    """
    deadline = time.time() + time_limit
    best_blocks = None
    best_count = upper_bound  # we want < upper_bound
    
    # Lower bound function
    def lower_bound(uncovered_mask, blocks_used):
        # LB1: ceil(uncovered_count / 35)
        cnt = uncovered_mask.bit_count()
        lb1 = (cnt + 34) // 35
        # LB2: pair packing bound. For each pair {a,b}, let u(a,b) = number of
        # uncovered t-sets containing pair (a,b). A block containing pair (a,b)
        # covers at most (k-2) = 5 such t-sets (the t-sets are (a,b,x) for x in block\{a,b},
        # but actually a block of size k containing a,b covers exactly (k-2) t-sets of the form (a,b,x)
        # where x is in the block. Wait: a t-set is 3 elements. If a block contains a,b and x,
        # then (a,b,x) is covered. So a block containing pair (a,b) covers exactly (k-2) t-sets
        # that contain pair (a,b), namely those with the third element in the block.
        # So we need at least ceil(u(a,b) / (k-2)) blocks containing pair (a,b).
        # But a single block can contain multiple pairs. This is a packing bound.
        # Simple version: sum over all pairs of ceil(u(a,b)/(k-2)) / (number of pairs per block)
        # Each block contains C(k,2)=21 pairs. So total "pair-covering capacity" per block is 21.
        # Sum of u(a,b) over all pairs = sum over uncovered t-sets of C(3,2)=3 = 3*cnt.
        # So LB2 = ceil(3*cnt / (21*(k-2))) = ceil(3*cnt / 105) = ceil(cnt/35). Same as LB1.
        # Better: for each pair, need ceil(u(a,b)/5) blocks. Sum these, divide by 21.
        # But this overcounts since one block covers multiple pairs.
        # Let's just use LB1 for now, plus a simple check.
        return max(lb1, 0)
    
    # MRV: find the uncovered t-set with fewest candidate blocks? 
    # Actually, we build blocks one at a time. The "variable" is which block to add next.
    # Better: pick an uncovered t-set, and branch on which block (containing that t-set) to add.
    # To limit branching, we only consider blocks that cover many uncovered t-sets.
    
    def solve(uncovered_mask, blocks, depth):
        nonlocal best_blocks, best_count
        if time.time() > deadline:
            return
        if uncovered_mask == 0:
            if depth < best_count:
                best_count = depth
                best_blocks = list(blocks)
            return
        if depth >= best_count:
            return
        lb = lower_bound(uncovered_mask, depth)
        if depth + lb >= best_count:
            return
        
        # Pick an uncovered t-set to "fix" - choose one with fewest options? 
        # Or just pick the first one. For efficiency, pick the one that appears in
        # the fewest potential blocks (but we don't have a fixed block set).
        # Instead: pick an uncovered t-set, and generate candidate blocks containing it.
        # To limit candidates, we only consider blocks that cover at least some threshold.
        
        # Get an uncovered t-set
        low = uncovered_mask & (-uncovered_mask)
        idx = low.bit_length() - 1
        ts = _ALL_TSETS[idx]
        ts_set = set(ts)
        
        # Generate candidate blocks: all 7-sets containing ts.
        # There are C(24-3, 7-3) = C(21,4) = 5985 such blocks. Too many to try all.
        # Instead, greedily build a few good blocks containing ts.
        # Or: use a heuristic to pick a good block containing ts.
        
        # Strategy: build a block containing ts by adding points one at a time,
        # maximizing gain. Try a few variants.
        candidates = []
        for variant in range(3):  # 3 variants
            block = list(ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for S in itertools.combinations(block, t - 1):
                        if x in S:
                            continue
                        tset = tuple(sorted(S + (x,)))
                        tidx = _TS_TO_IDX[tset]
                        if (uncovered_mask >> tidx) & 1:
                            gain += 1
                    if variant == 1:
                        gain += rng.random() * 0.5
                    elif variant == 2:
                        gain += rng.random() * 1.0
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                candidates.append(block)
        
        # Also try a few random blocks containing ts
        for _ in range(2):
            rest = [x for x in range(v) if x not in ts_set]
            if len(rest) >= k - t:
                block = list(ts) + sorted(rng.sample(rest, k - t))
                block.sort()
                candidates.append(block)
        
        # Sort candidates by gain (descending)
        def gain_of(block):
            g = 0
            bset = set(block)
            for ts2 in itertools.combinations(block, t):
                tidx = _TS_TO_IDX[tuple(sorted(ts2))]
                if (uncovered_mask >> tidx) & 1:
                    g += 1
            return g
        
        candidates.sort(key=gain_of, reverse=True)
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        for cand in unique_candidates:
            if time.time() > deadline:
                return
            # Compute new uncovered
            new_uncovered = uncovered_mask
            for ts2 in itertools.combinations(cand, t):
                tidx = _TS_TO_IDX[tuple(sorted(ts2))]
                new_uncovered &= ~(1 << tidx)
            if new_uncovered == uncovered_mask:
                continue  # no progress
            blocks.append(cand)
            solve(new_uncovered, blocks, depth + 1)
            blocks.pop()
            if depth + 1 >= best_count:
                break
    
    # Initial state
    full_mask = (1 << _N_TSETS) - 1
    solve(full_mask, [], 0)
    
    return best_blocks, best_count


def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # safety margin for 150s kill
    
    # Phase 1: Greedy to get initial upper bound
    greedy_time = min(15.0, total_budget * 0.15)
    rng = random.Random(42)
    greedy_blocks = _greedy_cover(v, k, t, rng, greedy_time)
    if not _is_valid(greedy_blocks, v, k, t):
        # Extend greedy to be valid
        deadline_g = start_time + total_budget * 0.25
        uncovered = (1 << _N_TSETS) - 1
        for b in greedy_blocks:
            bset = set(b)
            for ts in itertools.combinations(b, t):
                tidx = _TS_TO_IDX[tuple(sorted(ts))]
                uncovered &= ~(1 << tidx)
        while uncovered and time.time() < deadline_g:
            low = uncovered & (-uncovered)
            idx = low.bit_length() - 1
            ts = _ALL_TSETS[idx]
            block = list(ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for S in itertools.combinations(block, t - 1):
                        if x in S:
                            continue
                        tset = tuple(sorted(S + (x,)))
                        tidx2 = _TS_TO_IDX[tset]
                        if (uncovered >> tidx2) & 1:
                            gain += 1
                    gain += rng.random() * 0.01
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                greedy_blocks.append(block)
                for ts2 in itertools.combinations(block, t):
                    tidx2 = _TS_TO_IDX[tuple(sorted(ts2))]
                    uncovered &= ~(1 << tidx2)
    
    best_blocks = greedy_blocks
    best_count = len(greedy_blocks)
    
    # Phase 2: Backtracking search to improve
    remaining = total_budget - (time.time() - start_time) - 10.0  # 10s margin
    if remaining > 5.0:
        rng2 = random.Random(123)
        improved, improved_count = _backtrack_cover(v, k, t, best_count, remaining, rng2)
        if improved is not None and improved_count < best_count:
            best_blocks = improved
            best_count = improved_count
    
    # Final safety check
    if time.time() - start_time > total_budget - 5.0:
        return best_blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
