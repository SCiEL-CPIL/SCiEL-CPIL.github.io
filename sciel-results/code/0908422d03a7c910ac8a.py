"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

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

_NUM_FACTORS = [None] + [_prime_factors(i) for i in range(1, _N + 1)]

# For each prime, list of numbers (2..256) that have that prime as a factor
_PRIME_TO_NUMS = {p: [] for p in PRIMES}
for n in range(2, _N + 1):
    for p in _NUM_FACTORS[n]:
        _PRIME_TO_NUMS[p].append(n)

# Precompute for each number: its value and prime factors
# Sort numbers by value descending for search ordering

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # We'll use branch-and-bound on the set of free primes.
    # State: frozenset of free primes, current sum, current selected set
    # Goal: maximize sum
    
    # Upper bound function: given a set of free primes, what's the max additional sum?
    # For each free prime, the best number we can still form. But numbers can use multiple primes.
    # Simple upper bound: for each free prime p, max value of any number using only free primes and containing p.
    # This overcounts but is a valid upper bound if we take max over all such numbers... 
    # Actually no. Let me think differently.
    # 
    # Better upper bound: The sum of, for each free prime, the maximum "marginal value" we can get.
    # But this is hard. Let's use: sum over free primes of max{n in candidates : n uses only free primes and p in factors(n)}
    # This is an overcount because a number using 2 primes gets counted twice.
    # 
    # Even simpler valid upper bound: sum of all numbers that use only free primes.
    # This is valid (we can't do better than taking all of them) but loose.
    #
    # Let me use a tighter bound: for each free prime p, let best_p = max value of number using only free primes and containing p.
    # Then upper bound = sum of best_p over free primes, but divided by... no.
    #
    # Actually, the simplest tight bound: for each number that uses only free primes, we can take at most one number per prime.
    # The maximum weight independent set in the conflict graph restricted to free primes.
    # For an upper bound, we can use: for each free prime, the max number containing it (using only free primes).
    # Sum these, but each number with k primes is counted k times, so divide by... no, this isn't clean.
    #
    # Let me just use: sum of all numbers using only free primes as the upper bound. It's loose but valid.
    # And use strong pruning via the search order.
    
    # Alternative approach: since the problem is small, use a different formulation.
    # Each prime must be assigned to at most one number. Numbers with multiple primes require all those primes to be assigned to it.
    # 
    # Let me use a recursive search over primes, ordered by constraint.
    
    # Precompute: for each subset of primes... no, too many.
    
    # Let me try a different approach: local search from a good initial solution, with simulated annealing or tabu.
    
    # Actually, let me try a cleaner branch-and-bound:
    # - Order primes by number of candidate numbers (ascending) - most constrained first
    # - At each prime, branch over which number gets it (or none)
    # - Prune if current_sum + upper_bound < best_found
    
    # Upper bound: for remaining free primes, sum of max number per prime (overcounting multi-prime numbers)
    # This is valid as an upper bound because any feasible solution's contribution from remaining primes is <= this sum.
    
    # Let me precompute for quick access
    # For a given set of free primes (as a frozenset), compute upper bound
    # This is expensive to compute each time. Let me cache or use a simpler bound.
    
    # Simpler: upper bound = sum of (max number using only free primes and containing p) for each free prime p
    # Precompute for each prime p and each subset... no.
    
    # Let me just do a well-optimized local search instead.
    
    # Phase 1: Construct a good initial solution
    # For each prime, pick the best number. But numbers can share primes.
    # Greedy: sort numbers by value desc, pick if no conflict.
    
    def greedy_initial():
        used = set()
        selected = [1]
        total = 1
        for n in range(_N, 1, -1):
            pf = _NUM_FACTORS[n]
            if pf.isdisjoint(used):
                used |= pf
                selected.append(n)
                total += n
        return selected, total, used
    
    selected, total, used = greedy_initial()
    best_solution = list(selected)
    best_total = total
    
    # Phase 2: Local search with swaps
    # Try to improve by:
    # 1. Swapping one number for another
    # 2. Swapping one number for two numbers (if they use disjoint primes)
    # 3. Swapping two numbers for three numbers
    # 4. Adding a number if primes are free
    # 5. Removing a number to free primes for better ones
    
    selected_set = set(selected)
    
    def get_used_primes(sel):
        u = set()
        for s in sel:
            if s > 1:
                u |= _NUM_FACTORS[s]
        return u
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        all_used = get_used_primes(selected)
        
        # Try single swaps: remove a, add b where b > a and b's primes are free after removing a
        for a in list(selected):
            if a == 1:
                continue
            freed = _NUM_FACTORS[a]
            remaining = all_used - freed
            for b in range(_N, a, -1):  # only consider b > a for improvement
                if b in selected_set:
                    continue
                pf_b = _NUM_FACTORS[b]
                if pf_b.isdisjoint(remaining):
                    # This swap improves
                    selected.remove(a)
                    selected_set.remove(a)
                    selected.append(b)
                    selected_set.add(b)
                    best_total += b - a
                    improved = True
                    all_used = get_used_primes(selected)
                    break  # restart the loop with new state
        
        if improved:
            continue
        
        # Try adding a number with free primes
        free_primes = PRIME_SET - all_used
        if free_primes:
            for n in range(_N, 1, -1):
                if n in selected_set:
                    continue
                pf = _NUM_FACTORS[n]
                if pf.issubset(free_primes):
                    selected.append(n)
                    selected_set.add(n)
                    best_total += n
                    all_used |= pf
                    free_primes -= pf
                    improved = True
        
        if improved:
            continue
        
        # Try two-for-one swaps: remove a, add b and c where b,c > a combined
        # b and c must have disjoint primes, and both must be free after removing a
        for a in list(selected):
            if a == 1:
                continue
            freed = _NUM_FACTORS[a]
            remaining = all_used - freed
            # Find all numbers that can be added given remaining
            addable = []
            for n in range(_N, 1, -1):
                if n in selected_set or n == a:
                    continue
                pf_n = _NUM_FACTORS[n]
                if pf_n.isdisjoint(remaining):
                    addable.append(n)
            
            # Try pairs from addable
            for i in range(len(addable)):
                if time.time() > deadline:
                    break
                b = addable[i]
                pf_b = _NUM_FACTORS[b]
                for j in range(i + 1, len(addable)):
                    c = addable[j]
                    pf_c = _NUM_FACTORS[c]
                    if pf_b.isdisjoint(pf_c):
                        if b + c > a:
                            selected.remove(a)
                            selected_set.remove(a)
                            selected.append(b)
                            selected.append(c)
                            selected_set.add(b)
                            selected_set.add(c)
                            best_total += b + c - a
                            improved = True
                            all_used = get_used_primes(selected)
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try one-for-two swaps: remove a and b, add c where c > a + b
        # This is harder. Let's try: for each pair (a,b) in selected, see if we can add a single c
        # that uses primes from both a and b
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            a = selected[i]
            if a == 1:
                continue
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if b == 1:
                    continue
                freed = _NUM_FACTORS[a] | _NUM_FACTORS[b]
                remaining = all_used - freed
                for c in range(_N, 1, -1):
                    if c in selected_set:
                        continue
                    pf_c = _NUM_FACTORS[c]
                    if pf_c.isdisjoint(remaining):
                        if c > a + b:
                            selected.remove(a)
                            selected.remove(b)
                            selected_set.remove(a)
                            selected_set.remove(b)
                            selected.append(c)
                            selected_set.add(c)
                            best_total += c - a - b
                            improved = True
                            all_used = get_used_primes(selected)
                            break
                if improved:
                    break
            if improved:
                break
    
    # Final phase: try a more systematic improvement
    # For each prime, consider all numbers that could be assigned to it
    # and do a local optimization
    
    # Verify and return
    result = sorted(set(best_solution) if False else selected)
    # Deduplicate just in case
    result = sorted(set(result))
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback
                return [1] + [a for a in range(2, _N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
