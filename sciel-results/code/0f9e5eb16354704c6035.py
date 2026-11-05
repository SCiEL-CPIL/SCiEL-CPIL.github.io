"""Rewritten program: model-and-optimize approach for smooth-antichain representations.

Strategy:
  1. Precompute all smooth numbers up to 128.
  2. Precompute pairwise divisibility conflict mask (bitsets) for fast checking.
  3. For each target n (1..128), search for the minimum k (1..30) distinct smooth terms
     that sum to n and form a divisibility antichain.
  4. Use iterative deepening DFS with strong pruning:
     - Sum bounds (min/max possible remaining sums)
     - Conflict bitmask check against already-selected terms
     - Ordering by value to enable tight bounds
  5. Optimize total term count: for each target, find minimal k.
     To further optimize, we process targets in order of difficulty and cache results.
  6. Deadline guard: if time runs out, fall back to a fast greedy heuristic for remaining targets.

The key insight for "model-and-optimize": we parameterize the solution as a selection
of k indices into the sorted smooth list, and the objective is to minimize sum of k
across all targets. We search per-target for minimum k, which directly minimizes the
global objective since targets are independent.
"""

import time
import sys

# --- Smooth number generation ---
def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}

# --- Precompute divisibility conflict masks ---
# conflict_mask[i] is a bitmask where bit j is set if _SMOOTH[i] and _SMOOTH[j] conflict
# (i.e., one divides the other). We exclude self.
def _build_conflict_masks():
    masks = [0] * _N
    for i in range(_N):
        vi = _SMOOTH[i]
        m = 0
        for j in range(_N):
            if i == j:
                continue
            vj = _SMOOTH[j]
            # Check if vi divides vj or vj divides vi
            if vj % vi == 0 or vi % vj == 0:
                m |= (1 << j)
        masks[i] = m
    return masks

_CONFLICT = _build_conflict_masks()

# For each smooth index, precompute list of incompatible indices (for faster iteration)
_INCOMPAT = [bin(m).count('1') for m in _CONFLICT]  # just for counting, not used directly

# Global cache for solved (n, k) pairs
_solve_cache = {}

# --- Precompute prefix sums of sorted smooth values for bounding ---
# For a given starting index `start` and needing `k` more elements,
# the minimum possible sum is sum of the k smallest available >= _SMOOTH[start]
# The maximum possible sum is sum of the k largest available <= _SMOOTH[stop]
# But we search in descending order for better pruning, so let's think differently.

# We'll search by selecting values. To get good bounds, sort candidates descending.
# For a fixed target n, we only consider smooth values <= n.

def _solve_target(n, k):
    """Find k distinct smooth values summing to n, pairwise non-dividing. Returns list or None."""
    cache_key = (n, k)
    if cache_key in _solve_cache:
        return _solve_cache[cache_key]
    
    # Candidate values: smooth numbers <= n, sorted descending for better pruning
    cands = [v for v in _SMOOTH if v <= n]
    cands.sort(reverse=True)
    m = len(cands)
    if m < k:
        _solve_cache[cache_key] = None
        return None
    if k == 0:
        result = [] if n == 0 else None
        _solve_cache[cache_key] = result
        return result
    if k == 1:
        result = [n] if n in _VAL_INDEX else None
        _solve_cache[cache_key] = result
        return result
    
    # Build conflict mask in terms of cands indices using global _CONFLICT
    # Map each cand value to its global smooth index
    val_to_global = {v: i for i, v in enumerate(_SMOOTH)}
    cand_global = [val_to_global[v] for v in cands]
    
    # Build cand_conflict[i] as bitmask over cands indices
    cand_conflict = [0] * m
    for i in range(m):
        gi = cand_global[i]
        mask = 0
        for j in range(m):
            if i == j:
                continue
            gj = cand_global[j]
            if (_CONFLICT[gi] >> gj) & 1:
                mask |= (1 << j)
        cand_conflict[i] = mask
    
    # Precompute suffix sums for upper bound: sum of last k elements from position i
    # suffix_max[i][k] = sum of cands[i], cands[i+1], ..., cands[i+k-1] (the k largest from i onward)
    # Actually since sorted descending, cands[i] >= cands[i+1] >= ...
    # Max sum of k elements starting from index i is sum(cands[i:i+k])
    # Min sum of k elements starting from index i is... the k smallest available, which are at the end.
    # For pruning, we need:
    #   Upper bound on remaining sum: if we're at position i, the max we can add with r more elements
    #     is sum(cands[i:i+r]) since they're the largest remaining.
    #   Lower bound on remaining sum: the r smallest distinct positive integers? No, they must be from cands.
    #     The r smallest from cands[i:] are cands[m-r:m] but we can only pick from i onward.
    #     Actually, the minimum sum of r distinct elements from cands[i:] is sum of the r smallest in cands[i:],
    #     which are the last r elements of cands[i:]: cands[m-r:m] if m-i >= r, but that's global.
    #     Simpler: min sum of r distinct positive ints >= 1 is r(r+1)/2, but our values are specific.
    #     Use: min sum of r elements from cands[i:] is sum(cands[m-r:m]) only if m-r >= i.
    #     Let's just use a simple lower bound: r (each term >= 1).

    # Let's use:
    #   upper[i][r] = sum of cands[i : i+r]  (max possible sum of r elements from index i onward)
    #   lower[i][r] = sum of the r smallest in cands[i:] = sum(cands[m-r:m]) if (m - r) >= i else None
    #     But actually the r smallest in cands[i:] are cands[m-r:m] only if m-r >= i.
    #     If m-i < r, impossible.
    #     Otherwise, the r smallest in cands[i:] are cands[m-r:m] since array is descending.
    #     Wait no: cands is descending, so cands[i:] is also descending. The r smallest in cands[i:] 
    #     are the last r elements of cands[i:], i.e., cands[m-r:m] if m-r >= i, else the last m-i elements.
    #     Actually: cands[i:] has length m-i. The r smallest are cands[(m-i)-r + i : m] = cands[m-r:m].
    #     This is valid only if m-r >= i, i.e., r <= m-i.

    # Precompute upper bound table
    # upper[i][r] for r in 0..min(k, m-i)
    # We'll compute on the fly for speed.

    chosen = []
    chosen_mask = 0  # bitmask of selected cands indices

    def dfs(start, r, rem):
        """Select r more elements from cands[start:] summing to rem, avoiding conflicts with chosen_mask."""
        nonlocal chosen_mask
        if r == 0:
            return rem == 0
        if start + r > m:
            return False
        # Upper bound: max sum of r elements from cands[start:]
        if r > m - start:
            return False
        # Max possible sum: sum of the r largest elements from cands[start:], i.e., cands[start:start+r]
        max_sum = sum(cands[start:start+r])
        if rem > max_sum:
            return False
        # Min possible sum: sum of the r smallest elements from cands[start:]
        # cands is descending, so cands[start:] is descending.
        # The r smallest in cands[start:] are cands[m-r:m] only if m-r >= start.
        # If m-r < start, then we need the last r elements of cands[start:], which are cands[m-r:m] 
        # but that's wrong. Let me think again.
        # cands[start:] has elements cands[start], cands[start+1], ..., cands[m-1].
        # The r smallest of these are the last r elements: cands[m-r:m] if m-r >= start, 
        # otherwise all of cands[start:] has fewer than r elements (already checked).
        # So: the r smallest in cands[start:] are cands[m-r:m] when m-r >= start.
        # But wait, if start > m-r, then cands[m-r:m] includes elements before start.
        # Actually m-r >= start is equivalent to r <= m-start, which we already checked.
        # So when we reach here, m-r >= start, and the r smallest in cands[start:] are cands[m-r:m].
        min_sum = sum(cands[m-r:m])
        if rem < min_sum:
            return False
        
        for i in range(start, m - r + 1):
            v = cands[i]
            if v > rem:
                continue
            # Check conflict with already chosen
            if (cand_conflict[i] & chosen_mask) != 0:
                continue
            chosen.append(v)
            chosen_mask |= (1 << i)
            if dfs(i + 1, r - 1, rem - v):
                return True
            chosen.pop()
            chosen_mask &= ~(1 << i)
        return False
    
    result = list(chosen) if dfs(0, k, n) else None
    _solve_cache[cache_key] = result
    return result

# --- Fast fallback: greedy with backtracking for small k ---
def _greedy_solve(n, max_k=5):
    """Fast solution for target n, trying k=1,2,3,4,5."""
    for k in range(1, max_k + 1):
        result = _solve_target(n, k)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Leave 5s margin before 90s kill
    
    answers = [None] * 129
    
    # Process targets in order of difficulty: non-smooth first (they need more terms)
    # Smooth targets are easy (k=1)
    target_order = list(range(1, 129))
    # Put smooth numbers first (they're quick to solve)
    smooth_targets = [n for n in target_order if n in _VAL_INDEX]
    non_smooth_targets = [n for n in target_order if n not in _VAL_INDEX]
    target_order = smooth_targets + non_smooth_targets
    
    for n in target_order:
        if time.time() - start_time > DEADLINE:
            break
        
        # Try increasing k
        for k in range(1, 31):
            if time.time() - start_time > DEADLINE:
                break
            if k > n:  # Can't have more terms than the target value (each term >= 1)
                break
            
            result = _solve_target(n, k)
            if result is not None:
                answers[n] = result
                break
    
    # Handle any remaining unsolved targets with a fast fallback
    for n in range(1, 129):
        if answers[n] is None:
            if time.time() - start_time > DEADLINE - 2.0:
                # Last resort: try k up to 6 quickly
                for k in range(1, 7):
                    result = _solve_target(n, k)
                    if result is not None:
                        answers[n] = result
                        break
                if answers[n] is None:
                    # Emergency: two-term search
                    for i in range(len(_SMOOTH)):
                        a = _SMOOTH[i]
                        if a >= n:
                            break
                        b = n - a
                        if b <= 0:
                            continue
                        if b in _VAL_INDEX:
                            # Check divisibility
                            if a % b != 0 and b % a != 0:
                                answers[n] = [a, b]
                                break
                    if answers[n] is None:
                        # Should not happen for n<=128, but defensive
                        answers[n] = [1] if n == 1 else [2, n-2] if n-2 in _VAL_INDEX and (2 % (n-2) != 0 and (n-2) % 2 != 0) else [n]
            else:
                # Try up to k=8
                for k in range(1, 9):
                    result = _solve_target(n, k)
                    if result is not None:
                        answers[n] = result
                        break
                if answers[n] is None:
                    # Broader search
                    for k in range(9, 16):
                        result = _solve_target(n, k)
                        if result is not None:
                            answers[n] = result
                            break
                if answers[n] is None:
                    # Final fallback
                    for k in range(16, 31):
                        result = _solve_target(n, k)
                        if result is not None:
                            answers[n] = result
                            break
                    if answers[n] is None:
                        answers[n] = [n] if n in _VAL_INDEX else [1, n-1] if n-1 in _VAL_INDEX and (1 % (n-1) != 0 and (n-1) % 1 != 0) else [n]
    
    return answers[1:129]
