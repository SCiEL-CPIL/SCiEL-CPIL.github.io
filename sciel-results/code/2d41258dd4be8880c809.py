"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Uses efficient divisor generation and a best-first local search over
prime exponent configurations to find the optimal structure.
"""
import math
import time
import heapq
import itertools
from collections import Counter


def _tau_perp_fast(primes, exponents):
    """Compute tau_perp(n) given its prime factorization efficiently.
    
    Args:
        primes: list of distinct primes
        exponents: list of corresponding exponents
    
    Returns:
        Number of adjacent coprime pairs in the sorted divisor list.
    """
    k = len(primes)
    if k == 0:
        return 0
    
    # Generate all divisors as (value, mask) where mask tracks which primes divide it
    # For efficiency, use the fact that divisor d has prime set = {i : exp_i(d) > 0}
    # Two divisors are coprime iff their prime sets are disjoint
    
    # Build divisors incrementally
    divs = [(1, 0)]  # (value, bitmask of primes present)
    for i in range(k):
        p = primes[i]
        e = exponents[i]
        new_divs = []
        # Precompute powers
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        for (d, mask) in divs:
            for j, pp in enumerate(powers):
                new_mask = mask | (1 << i) if j > 0 else mask
                new_divs.append((d * pp, new_mask))
        divs = new_divs
    
    # Sort by value
    divs.sort(key=lambda x: x[0])
    
    count = 0
    for i in range(len(divs) - 1):
        if (divs[i][1] & divs[i + 1][1]) == 0:
            count += 1
    return count


def _tau_perp_squarefree(primes):
    """Optimized tau_perp for square-free n (all exponents = 1)."""
    k = len(primes)
    # Divisors are all subsets, sorted by value
    # Build all divisors with their prime masks
    divs = [(1, 0)]
    for i in range(k):
        p = primes[i]
        new_divs = []
        for (d, mask) in divs:
            new_divs.append((d, mask))
            new_divs.append((d * p, mask | (1 << i)))
        divs = new_divs
    
    divs.sort(key=lambda x: x[0])
    
    count = 0
    for i in range(len(divs) - 1):
        if (divs[i][1] & divs[i + 1][1]) == 0:
            count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Extended prime list up to 97 for better search coverage
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 1
    best_score = 0
    best_factors = None  # Store (primes, exponents) for best candidate
    
    def try_candidate(primes_list, exponents_list):
        nonlocal best_n, best_score, best_factors
        n = 1
        for p, e in zip(primes_list, exponents_list):
            n *= p ** e
        if 2 <= n <= 10**18:
            if all(e == 1 for e in exponents_list):
                score = _tau_perp_squarefree(primes_list)
            else:
                score = _tau_perp_fast(primes_list, exponents_list)
            if score > best_score:
                best_score = score
                best_n = n
                best_factors = (list(primes_list), list(exponents_list))
            return score
        return 0
    
    # Phase 1: Square-free with 15 primes (primorial)
    p15 = base_primes[:15]
    prod15 = 1
    for p in p15:
        prod15 *= p
    if prod15 <= 10**18:
        try_candidate(p15, [1] * 15)
    
    # Phase 2: Best-first search over prime sets (square-free, 15 primes)
    if time.time() > deadline:
        return best_n
    
    def get_prime_set_score(prime_tuple):
        """Score for a square-free set of primes."""
        prod = 1
        for p in prime_tuple:
            prod *= p
        if prod > 10**18:
            return -1
        return _tau_perp_squarefree(list(prime_tuple))
    
    # Initialize with first 15 primes
    initial_set = tuple(base_primes[:15])
    init_score = get_prime_set_score(initial_set)
    
    # Best-first search with swap moves
    # Priority queue: (-score, prime_set)
    visited = set()
    pq = [(-init_score, initial_set)]
    visited.add(initial_set)
    max_iterations = 8000
    
    iter_count = 0
    while pq and iter_count < max_iterations:
        if time.time() > deadline - 35:
            break
        neg_score, pset = heapq.heappop(pq)
        cur_score = -neg_score
        iter_count += 1
        
        # Generate neighbors: replace one prime with a different one
        used = set(pset)
        for i in range(15):
            # Try replacing pset[i] with a different prime not in the set
            for j in range(len(base_primes)):
                if j < 15:
                    # Try replacing with a smaller prime not already in set
                    new_p = base_primes[j]
                    if new_p in used or new_p == pset[i]:
                        continue
                else:
                    # Try replacing with a larger prime
                    new_p = base_primes[j]
                    if new_p in used or new_p == pset[i]:
                        continue
                new_set = list(pset)
                new_set[i] = new_p
                new_set.sort()
                new_tuple = tuple(new_set)
                if new_tuple in visited:
                    continue
                visited.add(new_tuple)
                
                # Quick product check
                prod = 1
                for p in new_tuple:
                    prod *= p
                    if prod > 10**18:
                        break
                if prod > 10**18:
                    continue
                
                ns = _tau_perp_squarefree(list(new_tuple))
                if ns > best_score:
                    best_score = ns
                    n_val = 1
                    for p in new_tuple:
                        n_val *= p
                    best_n = n_val
                    best_factors = (list(new_tuple), [1] * 15)
                
                heapq.heappush(pq, (-ns, new_tuple))
    
    # Phase 3: Try 14 primes with one squared
    if time.time() > deadline - 25:
        return best_n
    
    for sq_idx in range(10):
        if time.time() > deadline - 20:
            break
        sq_prime = base_primes[sq_idx]
        
        for start in range(0, 5):
            if time.time() > deadline - 20:
                break
            primes_list = []
            exponents_list = []
            prod = 1
            count = 0
            for i in range(start, start + 16):
                if i >= len(base_primes) or count >= 14:
                    break
                p = base_primes[i]
                e = 2 if p == sq_prime else 1
                if prod * p ** e > 10**18:
                    break
                primes_list.append(p)
                exponents_list.append(e)
                prod *= p ** e
                count += 1
            
            if len(primes_list) == 14:
                try_candidate(primes_list, exponents_list)
    
    # Phase 4: Try 13 primes with two squared
    if time.time() > deadline - 15:
        return best_n
    
    for sq1_idx in range(6):
        for sq2_idx in range(sq1_idx + 1, 10):
            if time.time() > deadline - 15:
                break
            sq1 = base_primes[sq1_idx]
            sq2 = base_primes[sq2_idx]
            
            for start in range(0, 4):
                if time.time() > deadline - 15:
                    break
                primes_list = []
                exponents_list = []
                prod = 1
                count = 0
                for i in range(start, start + 16):
                    if i >= len(base_primes) or count >= 13:
                        break
                    p = base_primes[i]
                    if p == sq1 or p == sq2:
                        e = 2
                    else:
                        e = 1
                    if prod * p ** e > 10**18:
                        break
                    primes_list.append(p)
                    exponents_list.append(e)
                    prod *= p ** e
                    count += 1
                
                if len(primes_list) == 13:
                    try_candidate(primes_list, exponents_list)
    
    # Phase 5: Polish - local search on exponent vectors for the best prime set
    if time.time() > deadline - 8:
        return best_n
    
    if best_factors is None:
        return best_n
    
    bp, be = best_factors
    
    # Try small perturbations on exponents
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(bp)):
            # Try incrementing one exponent
            if be[i] < 6:
                new_be = be[:]
                new_be[i] += 1
                prod = 1
                for p, e in zip(bp, new_be):
                    prod *= p ** e
                if prod <= 10**18:
                    old_score = best_score
                    try_candidate(bp, new_be)
                    if best_score > old_score:
                        be = new_be
                        improved = True
            # Try decrementing one exponent (if > 1)
            if be[i] > 1:
                new_be = be[:]
                new_be[i] -= 1
                old_score = best_score
                try_candidate(bp, new_be)
                if best_score > old_score:
                    be = new_be
                    improved = True
        # Try swapping: decrease one, increase another
        for i in range(len(bp)):
            if be[i] > 1:
                for j in range(len(bp)):
                    if i != j and be[j] < 6:
                        new_be = be[:]
                        new_be[i] -= 1
                        new_be[j] += 1
                        prod = 1
                        for p, e in zip(bp, new_be):
                            prod *= p ** e
                        if prod <= 10**18:
                            old_score = best_score
                            try_candidate(bp, new_be)
                            if best_score > old_score:
                                be = new_be
                                improved = True
    
    return best_n
