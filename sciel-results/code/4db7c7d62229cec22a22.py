"""Maximize count of distinct integers in 2..40 such that all 2^k subset products are distinct.

Strategy:
- The conflict condition is that no two distinct subsets have the same product.
- This is equivalent to: the exponent vectors (over all primes) of the selected numbers
  must be linearly independent over the non-negative integers (i.e., the only non-negative
  integer combination that gives zero is the trivial one).
- Equivalently, for any prime p, the sum of exponents of p in any subset must be unique
  across all subsets.
- A simpler sufficient condition: for each prime p, the set of p-exponent vectors of the
  selected numbers must be such that all subset-sums are distinct. This is guaranteed if
  the exponents of each prime form a set with distinct subset sums (e.g., powers of 2).

- Build a conflict graph: two numbers conflict if adding both would create a duplicate
  subset product. We use branch-and-bound / greedy with backtracking to find a large
  independent set.

- Key insight: primes >= 7 are always safe to include (their exponent is 1 in one prime
  and 0 in all others, so they act as independent dimensions).
- For small primes (2, 3, 5, 7), we can include composites whose exponent vectors
  are "independent" in the sense that all subset sums of exponents are distinct.

- We use a greedy + local search approach:
  1. Start with all primes in 2..40 (12 primes).
  2. Try to add composites one by one, checking validity.
  3. Use backtracking to explore alternative choices when stuck.
"""
import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 80.0

def _prime_factorization(n):
    """Return dict {prime: exponent} for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _all_subset_products(nums):
    """Return the set of all subset products of nums (including empty product 1)."""
    prods = {1}
    for x in nums:
        new_prods = set()
        for p in prods:
            new_prods.add(p * x)
        prods = prods | new_prods
    return prods

def _is_valid(nums):
    """Check if all 2^k subset products of nums are distinct."""
    if len(nums) == 0:
        return True
    prods = _all_subset_products(nums)
    return len(prods) == (1 << len(nums))

def _can_add(nums, prods, x):
    """Check if adding x to nums keeps all subset products distinct."""
    for p in prods:
        if (p * x) in prods:
            return False
    return True

def _try_greedy(candidates, initial=None):
    """Greedy: add candidates in given order if valid."""
    if initial is None:
        initial = []
    selected = list(initial)
    prods = _all_subset_products(selected)
    for x in candidates:
        if time.time() > _DEADLINE:
            break
        if _can_add(selected, prods, x):
            selected.append(x)
            for p in prods:
                prods.add(p * x)
    return selected

def _local_search(selected, all_numbers, max_iters=50):
    """Try to improve by swapping elements or adding new ones."""
    if time.time() > _DEADLINE:
        return selected
    selected = list(selected)
    used = set(selected)
    improved = True
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        improved = False
        # Try adding any unused number
        for x in all_numbers:
            if x in used:
                continue
            if _can_add(selected, _all_subset_products(selected), x):
                selected.append(x)
                used.add(x)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add two
        for i in range(len(selected)):
            if time.time() > _DEADLINE:
                break
            removed = selected[i]
            new_sel = selected[:i] + selected[i+1:]
            prods = _all_subset_products(new_sel)
            for x in all_numbers:
                if x in set(new_sel) or x == removed:
                    continue
                if _can_add(new_sel, prods, x):
                    new_sel2 = new_sel + [x]
                    prods2 = _all_subset_products(new_sel2)
                    # Try to add one more
                    for y in all_numbers:
                        if y in set(new_sel2) or y == removed:
                            continue
                        if _can_add(new_sel2, prods2, y):
                            selected = new_sel2 + [y]
                            used = set(selected)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    return selected

def _backtrack_search(candidates, current, prods, best, depth=0):
    """Branch and bound to find a large valid set."""
    if time.time() > _DEADLINE:
        return
    if len(current) > len(best):
        best[:] = current[:]
    if not candidates:
        return
    for i, x in enumerate(candidates):
        if time.time() > _DEADLINE:
            return
        if _can_add(current, prods, x):
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            current.append(x)
            _backtrack_search(candidates[i+1:], current, new_prods, best, depth+1)
            current.pop()
        # Also try skipping x
    return

def construct():
    global _DEADLINE
    if time.time() > _DEADLINE:
        return [2]
    
    all_numbers = list(range(2, 41))
    
    # Primes in 2..40
    primes = [n for n in range(2, 41) if all(n % d for d in range(2, int(n**0.5) + 1))]
    composites = [n for n in range(2, 41) if n not in primes]
    
    # Strategy 1: Start with primes, greedily add composites
    best = _try_greedy(composites, initial=list(primes))
    
    # Strategy 2: Try different orderings of composites
    for order in [sorted(composites), sorted(composites, reverse=True)]:
        if time.time() > _DEADLINE:
            break
        cand = _try_greedy(order, initial=list(primes))
        if len(cand) > len(best):
            best = cand
    
    # Strategy 3: Local search refinement
    if time.time() < _DEADLINE:
        best = _local_search(best, all_numbers)
    
    # Strategy 4: Backtrack on composites to see if we can do better
    if time.time() < _DEADLINE and len(best) < 16:
        # Use primes as base, try to add composites via backtracking
        base = list(primes)
        base_prods = _all_subset_products(base)
        remaining = [c for c in composites if c not in set(base)]
        current = list(base)
        bt_best = list(base)
        _backtrack_search(remaining, current, base_prods, bt_best)
        if len(bt_best) > len(best):
            best = bt_best
    
    # Final validation and cleanup
    best = [int(x) for x in best if 2 <= x <= 40]
    seen = set()
    out = []
    for x in best:
        if x not in seen:
            seen.add(x)
            out.append(x)
    
    if not out:
        out = [2]
    
    # Final validity check
    if not _is_valid(out):
        # Fallback to just primes
        out = [int(p) for p in primes]
    
    return list(out)
