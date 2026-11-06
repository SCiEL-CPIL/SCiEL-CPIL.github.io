"""Exact Maximum Independent Set for reciprocal-subsum-free sets in [1,24].

Structural approach:
  For a set S, for every a in S, 1/a != sum of any subset of {1/v : v in S, v != a}.

  This is a higher-order constraint (not pairwise), so we use branch-and-bound
  with incremental validity checks. The condition is hereditary: if S is valid,
  any subset is valid.

  Strategy:
    1. Start with a good initial solution (top half {13,...,24}).
    2. Greedy improvement: try adding elements one at a time.
    3. Local search: try remove-1-add-1 and remove-1-add-2 swaps.
    4. Branch-and-bound DFS with pruning to find exact maximum within time.
"""
import time
from fractions import Fraction
from itertools import combinations

DEADLINE = 85.0   # safety margin under 90s kill
_START = None


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`.
    Uses meet-in-the-middle for larger sets, direct check for small sets."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    if n <= 1:
        return any(v == target for v in vals)
    if n <= 4:
        # Direct enumeration for small sets
        for mask in range(1, 1 << n):
            s = Fraction(0)
            for i in range(n):
                if mask & (1 << i):
                    s += vals[i]
            if s == target:
                return True
        return False
    
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}

    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def is_valid(S):
    """Exact reciprocal-subsum-free check for a nonempty set S of ints."""
    S = list(set(S))
    if not S:
        return False
    r = {v: Fraction(1, v) for v in S}
    for a in S:
        others = [r[v] for v in S if v != a]
        if _subset_sum_possible(others, r[a]):
            return False
    return True


def _check_validity_incremental(current_set, new_elem):
    """Check if adding new_elem to current_set preserves validity.
    Only need to check:
    1. 1/new_elem is not a subsum of reciprocals of current_set
    2. For each a in current_set, 1/a is not a subsum of reciprocals of 
       (current_set - {a}) + {new_elem}
    """
    new_recip = Fraction(1, new_elem)
    current_recips = [Fraction(1, v) for v in current_set]
    
    # Check 1: new element's reciprocal vs. current set
    if _subset_sum_possible(current_recips, new_recip):
        return False
    
    # Check 2: each existing element's reciprocal vs. (others + new)
    for a in current_set:
        a_recip = Fraction(1, a)
        others = [Fraction(1, v) for v in current_set if v != a]
        others.append(new_recip)
        if _subset_sum_possible(others, a_recip):
            return False
    
    return True


def _greedy_add(best, all_elems):
    """Try to add elements greedily to best."""
    improved = True
    while improved and not _deadline_hit():
        improved = False
        for v in all_elems:
            if _deadline_hit():
                break
            if v in best:
                continue
            if _check_validity_incremental(best, v):
                best = best + [v]
                improved = True
                break
    return best


def _local_search(best):
    """Local search: try remove-1-add-1, remove-1-add-2, and remove-2-add-3 swaps."""
    all_elems = list(range(1, 25))
    improved = True
    while improved and not _deadline_hit():
        improved = False
        best_set = set(best)
        
        # Try remove-1-add-1 (net +0, only useful if we can then add more)
        # Skip this since it doesn't increase size directly
        
        # Try remove-1-add-2 (net +1)
        for i in range(len(best)):
            if _deadline_hit():
                break
            for j, k in combinations(all_elems, 2):
                if _deadline_hit():
                    break
                if j in best_set or k in best_set:
                    continue
                candidate = [x for x in best if x != best[i]] + [j, k]
                if len(candidate) > len(best) and is_valid(candidate):
                    best = candidate
                    best_set = set(best)
                    improved = True
                    break
            if improved:
                break
        
        if improved or _deadline_hit():
            continue
        
        # Try remove-2-add-3 (net +1)
        for i, j in combinations(range(len(best)), 2):
            if _deadline_hit():
                break
            for k, m, n in combinations(all_elems, 3):
                if _deadline_hit():
                    break
                if k in best_set or m in best_set or n in best_set:
                    continue
                candidate = [x for x in best if x != best[i] and x != best[j]] + [k, m, n]
                if len(candidate) > len(best) and is_valid(candidate):
                    best = candidate
                    best_set = set(best)
                    improved = True
                    break
            if improved:
                break
    
    return best


def _branch_and_bound(best, all_elems):
    """Branch-and-bound DFS to find exact maximum within time budget."""
    best = list(best)
    best_size = len(best)
    best_set = set(best)
    
    # Sort elements: try larger elements first (smaller reciprocals, less likely to conflict)
    order = sorted(all_elems, key=lambda x: -x)
    
    def dfs(current_set, idx, current_size):
        nonlocal best, best_size, best_set
        
        if _deadline_hit():
            return
        
        # Prune: if we can't beat best, stop
        if current_size + (len(order) - idx) <= best_size:
            return
        
        if idx >= len(order):
            return
        
        # Try including order[idx]
        elem = order[idx]
        if elem not in best_set:
            if _check_validity_incremental(current_set, elem):
                new_set = current_set + [elem]
                dfs(new_set, idx + 1, current_size + 1)
        
        # Try excluding order[idx]
        dfs(current_set, idx + 1, current_size)
    
    dfs(best, 0, len(best))
    return best


def construct():
    global _START
    _START = time.time()

    # Start with the known good seed: top half {13,...,24}
    best = list(range(13, 25))
    best_size = len(best)
    all_elems = list(range(1, 25))
    
    # Phase 1: Greedy improvement
    best = _greedy_add(best, all_elems)
    best_size = len(best)
    
    # Phase 2: Local search
    if not _deadline_hit():
        best = _local_search(best)
        best_size = len(best)
    
    # Phase 3: Branch-and-bound if time remains
    if not _deadline_hit():
        remaining_time = DEADLINE - (time.time() - _START)
        if remaining_time > 5:  # Only do B&B if we have enough time
            best = _branch_and_bound(best, all_elems)
            best_size = len(best)
    
    # Final validation
    if not is_valid(best):
        # Fallback to known good
        best = list(range(13, 25))
    
    return list(best)


if __name__ == "__main__":
    print(construct())
