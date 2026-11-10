"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from itertools import combinations

_N = 256

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

PRIMES = _sieve_primes(_N)
PRIME_SET = set(PRIMES)

def _prime_factors(n):
    if n == 1:
        return frozenset()
    factors = set()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return frozenset(factors)

# Precompute prime factors for all numbers
_NUM_FACTORS = [None] + [_prime_factors(i) for i in range(1, _N + 1)]

# Precompute conflict: for each number, which other numbers conflict with it
# Two numbers conflict if they share a prime factor
_CONFLICTS = [set() for _ in range(_N + 1)]
for i in range(2, _N + 1):
    for j in range(i + 1, _N + 1):
        if not _NUM_FACTORS[i].isdisjoint(_NUM_FACTORS[j]):
            _CONFLICTS[i].add(j)
            _CONFLICTS[j].add(i)

def construct():
    """
    Find a set of distinct integers from 1..256 such that every pair is coprime,
    maximizing their sum.
    
    Strategy: Branch and bound with strong upper bounds using prime-based relaxation.
    """
    deadline = time.time() + 78  # Safety margin before 90s kill
    
    # Always include 1
    # For numbers 2..256, we need to select a subset where no two share a prime factor.
    
    # Key insight: Each prime can be used by at most one selected number.
    # Upper bound: For each prime p, the best number that uses p (and is compatible)
    # gives an upper bound on the contribution of that prime.
    
    # Let's use a recursive branch-and-bound with memoization on available primes.
    # But 2^54 is too large, so we need a smarter approach.
    
    # Alternative: Use the fact that numbers have at most 4 prime factors.
    # We can do a search where at each step we pick a number, remove all conflicting
    # numbers, and recurse.
    
    # For a strong upper bound: 
    # - For each remaining prime p, let best[p] = max value of number n that uses p
    #   and is compatible with current selection
    # - Upper bound = current_sum + sum over all remaining primes p of best[p] / (number of primes in best number)
    # This is a relaxation.
    
    # Actually, a simpler and tighter bound:
    # For each prime p not yet used, the maximum additional value we can get is
    # the max over all numbers n that use p and are compatible of (n / len(primes(n)))
    # But this double-counts. 
    
    # Let's use a different approach: 
    # The problem is equivalent to maximum weight independent set in a graph.
    # We can use a greedy coloring bound.
    
    # Better: Use the structure that each number uses a subset of primes.
    # This is a set packing problem. The LP relaxation gives an upper bound.
    
    # For practical purposes, let's use:
    # 1. Greedy initialization with multiple orderings
    # 2. Local search with 1-swap and 2-swap moves
    # 3. Branch and bound for remaining time
    
    candidates = list(range(2, _N + 1))
    
    def greedy_select(order):
        """Select numbers in given order, skipping conflicts."""
        used_primes = set()
        selected = [1]
        total = 1
        for n in order:
            pf = _NUM_FACTORS[n]
            if not pf:
                continue
            if pf.isdisjoint(used_primes):
                used_primes |= pf
                selected.append(n)
                total += n
        return selected, total, used_primes
    
    # Try multiple orderings
    orderings = [
        sorted(candidates, key=lambda x: -x),
        sorted(candidates, key=lambda x: (-x / max(len(_NUM_FACTORS[x]), 1), -x)),
        sorted(candidates, key=lambda x: (len(_NUM_FACTORS[x]), -x)),
        sorted(candidates, key=lambda x: (-x / (len(_NUM_FACTORS[x]) + 0.5), -x)),
        sorted(candidates, key=lambda x: (-x, len(_NUM_FACTORS[x]))),
    ]
    
    best_selected = None
    best_total = 0
    best_used = None
    
    for ordering in orderings:
        sel, tot, used = greedy_select(ordering)
        if tot > best_total:
            best_total = tot
            best_selected = sel[:]
            best_used = used
    
    selected_set = set(best_selected)
    
    # Local search: 1-swap moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        all_used = set()
        for s in best_selected:
            if s > 1:
                all_used |= _NUM_FACTORS[s]
        
        for a in list(best_selected):
            if a == 1:
                continue
            freed = _NUM_FACTORS[a]
            remaining_used = all_used - freed
            
            best_b = None
            best_gain = 0
            for b in candidates:
                if b in selected_set:
                    continue
                pf_b = _NUM_FACTORS[b]
                if not pf_b:
                    continue
                if pf_b.isdisjoint(remaining_used):
                    gain = b - a
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b is not None and best_gain > 0:
                best_selected.remove(a)
                selected_set.remove(a)
                best_selected.append(best_b)
                selected_set.add(best_b)
                best_total += best_gain
                improved = True
                all_used = set()
                for s in best_selected:
                    if s > 1:
                        all_used |= _NUM_FACTORS[s]
    
    # Try adding more numbers if possible
    all_used = set()
    for s in best_selected:
        if s > 1:
            all_used |= _NUM_FACTORS[s]
    
    free_primes = PRIME_SET - all_used
    if free_primes:
        for n in sorted(candidates, key=lambda x: -x):
            if n in selected_set:
                continue
            pf = _NUM_FACTORS[n]
            if not pf:
                continue
            if pf.issubset(free_primes):
                best_selected.append(n)
                selected_set.add(n)
                best_total += n
                all_used |= pf
                free_primes -= pf
    
    # 2-swap local search: replace two selected numbers with two unselected numbers
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            all_used = set()
            for s in best_selected:
                if s > 1:
                    all_used |= _NUM_FACTORS[s]
            
            non_one_selected = [s for s in best_selected if s != 1]
            
            # Try all pairs of selected numbers to remove
            for i in range(len(non_one_selected)):
                for j in range(i + 1, len(non_one_selected)):
                    a = non_one_selected[i]
                    b = non_one_selected[j]
                    freed = _NUM_FACTORS[a] | _NUM_FACTORS[b]
                    remaining_used = all_used - freed
                    
                    # Find best pair of unselected numbers to add
                    best_pair = None
                    best_gain = 0
                    for c in candidates:
                        if c in selected_set:
                            continue
                        pf_c = _NUM_FACTORS[c]
                        if not pf_c:
                            continue
                        if not pf_c.isdisjoint(remaining_used):
                            continue
                        for d in candidates:
                            if d <= c or d in selected_set:
                                continue
                            pf_d = _NUM_FACTORS[d]
                            if not pf_d:
                                continue
                            if not pf_d.isdisjoint(remaining_used):
                                continue
                            if not pf_c.isdisjoint(pf_d):
                                continue
                            gain = c + d - a - b
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (c, d)
                    
                    if best_pair is not None and best_gain > 0:
                        best_selected.remove(a)
                        best_selected.remove(b)
                        selected_set.remove(a)
                        selected_set.remove(b)
                        best_selected.append(best_pair[0])
                        best_selected.append(best_pair[1])
                        selected_set.add(best_pair[0])
                        selected_set.add(best_pair[1])
                        best_total += best_gain
                        improved = True
                        all_used = set()
                        for s in best_selected:
                            if s > 1:
                                all_used |= _NUM_FACTORS[s]
                        break
                if improved:
                    break
    
    # Branch and bound for remaining time
    if time.time() < deadline - 5:
        # Try a limited branch and bound from the current best
        # Sort candidates by value descending
        sorted_candidates = sorted(candidates, key=lambda x: -x)
        
        def branch_and_bound(remaining, used_primes, current_sum, current_set, depth=0):
            """Recursive branch and bound."""
            nonlocal best_total, best_selected
            if time.time() > deadline - 3:
                return
            if not remaining:
                if current_sum > best_total:
                    best_total = current_sum
                    best_selected = current_set[:]
                return
            
            # Upper bound: current_sum + sum of all remaining that don't conflict
            # Quick bound: current_sum + sum of values of remaining
            ub = current_sum + sum(remaining)
            if ub <= best_total:
                return
            
            # Pick the first candidate
            n = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include n
            pf = _NUM_FACTORS[n]
            if pf.isdisjoint(used_primes):
                new_remaining = [x for x in rest if _NUM_FACTORS[x].isdisjoint(pf)]
                branch_and_bound(new_remaining, used_primes | pf, current_sum + n, current_set + [n], depth + 1)
            
            # Option 2: Exclude n
            branch_and_bound(rest, used_primes, current_sum, current_set, depth + 1)
        
        # Start B&B from scratch with a time limit
        initial_used = set()
        # Only include primes from numbers in current best that we might want to change
        # Actually, let's just try B&B on all candidates with a depth limit
        # This might be too slow, so let's limit it
        
        # For now, skip full B&B as it's likely too slow for 54 primes
        pass
    
    # Verify result
    result = sorted(best_selected)
    
    # Final verification: check pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [a for a in range(2, _N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Ensure all are Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
