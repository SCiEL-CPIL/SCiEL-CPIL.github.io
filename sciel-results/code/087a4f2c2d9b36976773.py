import time
import math
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

PRIMES = _sieve(1000)
N_PRIMES = len(PRIMES)

def _budget_ok(residues, k):
    """Exact reciprocal budget check for the first k selected primes."""
    total = Fraction(0, 1)
    for i in range(k):
        p = PRIMES[i]
        total += Fraction(1, p)
    return total <= 1

def _coverage_length(residues, k, limit=100000):
    """Compute L such that every integer in 1..L is covered by the first k classes."""
    if k == 0:
        return 0
    covered = bytearray(limit + 1)
    for i in range(k):
        p = PRIMES[i]
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        # mark
        for n in range(first, limit + 1, p):
            covered[n] = 1
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _build_counts(residues, k, limit):
    """For each n in 1..limit, count coverage classes covering n. Also returns gap list."""
    counts = [0] * (limit + 1)
    for i in range(k):
        p = PRIMES[i]
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        for n in range(first, limit + 1, p):
            counts[n] += 1
    return counts

def _compute_L_from_counts(counts):
    L = 0
    for i in range(1, len(counts)):
        if counts[i] > 0:
            L = i
        else:
            break
    return L

def _greedy_assign(prime_subset, limit):
    """
    Given a list of primes (indices into PRIMES), assign residues greedily
    to maximize coverage of 1..limit.
    """
    residues = [0] * len(prime_subset)
    # start with r=0 for all
    # compute initial L
    k = len(prime_subset)
    counts = _build_counts([0] * k, k, limit)
    L = _compute_L_from_counts(counts)
    for i in range(k):
        p = prime_subset[i]
        best_gain = -1
        best_r = residues[i]
        for r in range(p):
            if r == residues[i]:
                continue
            counts[r] = 0
            # undo old
            old_r = residues[i]
            old_first = p if old_r == 0 else old_r
            for n in range(old_first, limit + 1, p):
                counts[n] -= 1
            # apply new
            new_first = p if r == 0 else r
            for n in range(new_first, limit + 1, p):
                counts[n] += 1
            # recompute
            # find gap
            gap_pos = None
            new_L = 0
            for idx in range(1, limit + 1):
                if counts[idx] > 0:
                    new_L = idx
                else:
                    gap_pos = idx
                    break
            if gap_pos is None:
                new_L = limit
            gain = new_L - L
            if gain > best_gain:
                best_gain = gain
                best_r = r
                # record best counts? expensive. We'll just track L
                # actually just need gain, we revert
                # revert
                for n in range(new_first, limit + 1, p):
                    counts[n] -= 1
                for n in range(old_first, limit + 1, p):
                    counts[n] += 1
            else:
                # revert
                for n in range(new_first, limit + 1, p):
                    counts[n] -= 1
                for n in range(old_first, limit + 1, p):
                    counts[n] += 1
        residues[i] = best_r
        # recompute L
        counts = _build_counts(residues, k, limit)
        L = _compute_L_from_counts(counts)
    return residues, L

def _local_search_fast(residues, k, limit, time_budget):
    """
    Local search: try changing one residue to extend L. 
    We maintain counts array and gap list implicitly via first gap.
    """
    start_t = time.time()
    counts = _build_counts(residues, k, limit)
    L = _compute_L_from_counts(counts)
    best_L = L
    # find current gap
    gap = L + 1
    while True:
        if time.time() - start_t > time_budget:
            break
        if gap > limit:
            break
        # try to cover gap by changing some residue
        # We need some class to cover gap
        # find primes p in prime subset where gap % p != r_i currently
        # and where changing r_i to gap%p increases coverage
        changed = False
        # iterate over primes in random or reverse order? Let's do small primes first for efficiency
        for i in range(k):
            p = PRIMES[i]
            if p <= 2:
                continue
            current_r = residues[i]
            target_r = gap % p
            if target_r == current_r:
                continue
            # changing i to target_r will cover gap.
            # does it reduce L? 
            old_first = p if current_r == 0 else current_r
            new_first = p if target_r == 0 else target_r
            
            delta_change = 0
            # we need to see if new L (after covering gap) is > L
            # we don't know exact L without full scan. But if gap is the first gap, 
            # covering it might reveal next gap.
            # Let's just do the change and compute new L.
            # To avoid full scan, we can just do a local scan from L+1 upwards.
            
            # apply
            for n in range(old_first, limit + 1, p):
                counts[n] -= 1
            for n in range(new_first, limit + 1, p):
                counts[n] += 1
            
            # scan for new L
            new_L = L
            while new_L < limit and counts[new_L + 1] > 0:
                new_L += 1
            
            if new_L > best_L:
                best_L = new_L
                changed = True
                # accept move
                residues[i] = target_r
                L = new_L
                # recompute gap
                if L < limit:
                    gap = L + 1
                    break # restart search from this new gap
            else:
                # revert
                for n in range(new_first, limit + 1, p):
                    counts[n] -= 1
                for n in range(old_first, limit + 1, p):
                    counts[n] += 1
    return best_L

def _local_search_multi_start(primes_list, max_time=20.0):
    """
    Try different subsets and initial assignments.
    """
    start_t = time.time()
    best_L = 0
    best_residues = []
    best_subset = []
    
    # We use primes sorted by size? The budget constraint is sum 1/p <= 1.
    # The set of primes is determined by which we pick.
    # Strategy: Use a fixed set of small primes? 
    # The sum of reciprocals of primes < 100 is ~ 1.7. So we can't use all.
    # We need to select a subset with sum 1/p <= 1.
    # A known good covering system uses primes 2,3,5,7, etc. but carefully.
    # For now, let's fix the set of primes to the first K primes such that sum <= 1.
    # Then optimize residues.
    
    # Let's find max K
    k_max = 0
    total = 0.0
    for i, p in enumerate(PRIMES):
        if total + 1.0/p <= 1.0:
            total += 1.0/p
            k_max = i + 1
        else:
            break
    
    # k_max is around 15-20?
    # Sum 1/p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37 ~ 0.5+0.33+0.2+0.14+0.09+0.077+0.058+0.052+0.043+0.034+0.032+0.027 = 1.54 > 1.
    # So K is small. 
    # Actually, let's compute precisely.
    # This approach (fixed set, optimize residues) is good.
    # But maybe we can swap a prime for a larger one?
    # For now, stick to prefix primes.
    
    limit = 5000 # search limit for local search, can extend later
    
    # 1. Greedy assign for the fixed set
    primes_list = [PRIMES[i] for i in range(k_max)]
    residues, L = _greedy_assign(primes_list, limit)
    if L > best_L:
        best_L = L
        best_residues = residues
        best_subset = primes_list
    
    # 2. Local search
    L = _local_search_fast(residues, k_max, limit, max_time - (time.time() - start_t))
    if L > best_L:
        best_L = L
    
    return best_L, best_residues, best_subset

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety
    
    # 1. Determine the prime set
    # We want a set of primes with sum(1/p) <= 1 that maximizes coverage.
    # The prefix of primes is a natural candidate.
    k_prefix = 0
    exact_sum = Fraction(0, 1)
    for i, p in enumerate(PRIMES):
        if exact_sum + Fraction(1, p) <= 1:
            exact_sum += Fraction(1, p)
            k_prefix = i + 1
        else:
            break
    
    primes_used = [PRIMES[i] for i in range(k_prefix)]
    
    # 2. Initial assignment: Greedy
    # Start with a limit
    search_limit = 10000
    
    # We will run _greedy_assign and _local_search
    residues, L0 = _greedy_assign(primes_used, search_limit)
    
    # 3. Local search
    # _local_search_fast expects residues list and k
    k = k_prefix
    L = _local_search_fast(residues, k, search_limit, 60.0)
    
    # 4. Check if we can extend by adding more primes?
    # No, budget is fixed at 1.0.
    # Can we swap primes?
    # e.g. replace a large prime in the set with a different one?
    # The prefix primes are the smallest. Replacing with larger primes reduces density, likely hurting.
    # So we stick with the prefix.
    
    # 5. Final verification and return
    final_classes = []
    for i in range(k):
        p = PRIMES[i]
        r = residues[i]
        final_classes.append([p, r])
        
    # Verify budget
    exact_total = Fraction(0, 1)
    for p, r in final_classes:
        exact_total += Fraction(1, p)
    if exact_total > 1:
        # Should not happen
        # Remove largest prime?
        # Just fallback to a trivial valid set if this happens
        return [[2, 1], [3, 2]]
        
    # Calculate true coverage L
    # We calculated L during search, but let's be sure for the returned value
    # We don't need to return L, just the classes. The evaluator calculates L.
    # But we should ensure the returned classes are valid.
    
    return final_classes

# EVOLVE-BLOCK-END
