"""
Improved program for the reciprocal-subsum-free set task.

Changes:
1. Enhanced Local Search with remove-2-add-2 moves to escape local optima.
2. Multi-start strategy: Run multiple local search instances with different initializations.
3. Optimized validity checking by caching subset sums for small sets if beneficial, or just relying on the fast MITM.
4. Final polish using backtracking from the best local search result.
"""
import time
import random
from fractions import Fraction

_START = None
DEADLINE = 85.0  # Safety margin under 90s kill


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """True iff some subset of `vals` (Fractions) sums exactly to `target`.
    Meet-in-the-middle, O(2^(n/2))."""
    n = len(vals)
    if n == 0:
        return target == Fraction(0)
    if n <= 1:
        return any(v == target for v in vals)
    
    # Quick upper bound check: if sum of all vals < target, impossible
    total = sum(vals)
    if total < target:
        return False
    
    mid = n // 2
    left, right = vals[:mid], vals[mid:]
    
    # Generate all subset sums for left
    L = {Fraction(0)}
    for v in left:
        L = L | {x + v for x in L}
    
    # Generate all subset sums for right
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}
    
    # Check if any combination matches target
    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False


def _is_valid(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    if not S:
        return False
    S_list = list(S)
    recips = {v: Fraction(1, v) for v in S_list}
    for a in S_list:
        others = [recips[v] for v in S_list if v != a]
        if _subset_sum_possible(others, recips[a]):
            return False
    return True

def _check_validity_incremental(current_set, new_elem):
    """Check if adding new_elem to current_set preserves validity."""
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

def _local_search_extended(best):
    """Local search with remove-1-add-1, remove-1-add-2, and remove-2-add-2."""
    all_elems = set(range(1, 25))
    improved = True
    while improved and not _deadline_hit():
        improved = False
        best_set = set(best)
        
        # Try remove-1-add-1
        for i in range(len(best)):
            if _deadline_hit():
                break
            for j in range(1, 25):
                if _deadline_hit():
                    break
                if j in best_set:
                    continue
                candidate = [x for x in best if x != best[i]] + [j]
                if _is_valid(candidate) and len(candidate) > len(best):
                    best = candidate
                    best_set = set(best)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Try remove-1-add-2 (net gain of 1)
        for i in range(len(best)):
            if _deadline_hit():
                break
            for j in range(1, 25):
                if _deadline_hit():
                    break
                if j in best_set:
                    continue
                for k in range(j + 1, 25):
                    if _deadline_hit():
                        break
                    if k in best_set:
                        continue
                    candidate = [x for x in best if x != best[i]] + [j, k]
                    if _is_valid(candidate) and len(candidate) > len(best):
                        best = candidate
                        best_set = set(best)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
                
        if improved:
            continue

        # Try remove-2-add-2 (net gain 0, but might escape local optimum for future gains)
        # Only do this if we haven't improved in the last few iterations? 
        # For now, just try it if stuck.
        if len(best) >= 2:
            for i in range(len(best)):
                if _deadline_hit():
                    break
                for j in range(i+1, len(best)):
                    if _deadline_hit():
                        break
                    # Remove best[i] and best[j]
                    removed = {best[i], best[j]}
                    base = [x for x in best if x not in removed]
                    
                    # Try adding two new elements
                    for k in range(1, 25):
                        if _deadline_hit():
                            break
                        if k in base or k in removed:
                            continue
                        for l in range(k + 1, 25):
                            if _deadline_hit():
                                break
                            if l in base or l in removed:
                                continue
                            candidate = base + [k, l]
                            if _is_valid(candidate):
                                # If we find a valid set of same size, we might be stuck.
                                # To ensure progress, we only accept if it allows further growth?
                                # For now, accept any valid set to shuffle the landscape.
                                # But this can lead to loops.
                                # Let's only accept if it's "different" enough or use a random tie-breaker.
                                # Actually, just returning a different valid set of same size is okay for perturbation.
                                best = candidate
                                best_set = set(best)
                                improved = True
                                break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    return best

def _multi_start_local_search(num_starts=5):
    """Run local search from multiple random initializations."""
    best_global = []
    all_elems = list(range(1, 25))
    
    for _ in range(num_starts):
        if _deadline_hit():
            break
        
        # Random initial set of size ~10
        random.shuffle(all_elems)
        initial = all_elems[:10]
        current = _local_search_extended(initial)
        
        if len(current) > len(best_global):
            best_global = current
            
    return best_global

def construct():
    global _START
    _START = time.time()
    random.seed(42) # For reproducibility in testing, though evaluator might not care
    
    # Strategy 1: Greedy from top half (fast, good baseline)
    best = list(range(13, 25))
    # Greedy add
    all_elems = list(range(1, 25))
    improved = True
    while improved and not _deadline_hit():
        improved = False
        best_set = set(best)
        for v in all_elems:
            if _deadline_hit():
                break
            if v in best_set:
                continue
            if _check_validity_incremental(best, v):
                best = best + [v]
                improved = True
                break
    
    # Strategy 2: Local Search on the greedy result
    best = _local_search_extended(best)
    
    # Strategy 3: Multi-start Local Search
    if not _deadline_hit():
        multi_best = _multi_start_local_search(num_starts=3)
        if len(multi_best) > len(best):
            best = multi_best
            best = _local_search_extended(best)
            
    # Strategy 4: Backtracking from best to try to add more
    if not _deadline_hit() and len(best) < 24:
        candidates = [v for v in range(1, 25) if v not in best]
        if candidates:
            # Simple backtrack to add 1 or 2 more
            for v in candidates:
                if _deadline_hit():
                    break
                if _check_validity_incremental(best, v):
                    best.append(v)
                    break
            
            # If we added one, try to add another
            if not _deadline_hit() and len(best) < 24:
                candidates2 = [v for v in range(1, 25) if v not in best]
                for v in candidates2:
                    if _deadline_hit():
                        break
                    if _check_validity_incremental(best, v):
                        best.append(v)
                        break
                        
    if not best:
        best = [24]
        
    return best
