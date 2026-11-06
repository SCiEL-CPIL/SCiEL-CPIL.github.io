"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Uses efficient divisor generation and a best-first local search over
prime exponent configurations to find the optimal structure.
"""
import math
import time
import heapq
import itertools


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
    
    # More primes for flexibility
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                   127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                   179, 181, 191, 193, 197, 199]
    
    best_n = 1
    best_score = 0
    
    def try_candidate(primes_list, exponents_list):
        nonlocal best_n, best_score
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
            return score
        return 0
    
    # Phase 1: Square-free with 15 primes (primorial)
    p15 = base_primes[:15]
    prod15 = 1
    for p in p15:
        prod15 *= p
    if prod15 <= 10**18:
        try_candidate(p15, [1] * 15)
    
    # Phase 2: Local search / hill climbing on the set of primes
    # Start with 15 smallest primes, try swaps
    if time.time() > deadline:
        return best_n
    
    # Best-first search over prime sets (square-free, 15 primes)
    # State: tuple of 15 primes (sorted)
    # Moves: replace one prime with a larger unused prime, or replace two primes
    
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
    max_iterations = 5000
    
    iter_count = 0
    while pq and iter_count < max_iterations:
        if time.time() > deadline - 30:
            break
        neg_score, pset = heapq.heappop(pq)
        cur_score = -neg_score
        iter_count += 1
        
        # Generate neighbors: replace one prime with a different one
        used = set(pset)
        for i in range(15):
            # Try replacing pset[i] with a larger prime not in the set
            # Only try primes that are "close" to keep search manageable
            for j in range(15, min(45, len(base_primes))):
                new_p = base_primes[j]
                if new_p in used:
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
                
                heapq.heappush(pq, (-ns, new_tuple))
    
    # Phase 3: Try 14 primes with one squared
    if time.time() > deadline - 20:
        return best_n
    
    # For 14 primes with one squared, the total number of divisors is 3 * 2^13 = 24576
    # Try different choices for which prime to square and which 14 primes to use
    for sq_idx in range(8):
        if time.time() > deadline - 15:
            break
        sq_prime = base_primes[sq_idx]
        
        # Try using the first 14 primes (with sq_prime squared)
        # and variations
        for start in range(0, 4):
            if time.time() > deadline - 15:
                break
            primes_list = []
            exponents_list = []
            prod = 1
            count = 0
            for i in range(start, start + 15):
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
    if time.time() > deadline - 10:
        return best_n
    
    for sq1_idx in range(5):
        for sq2_idx in range(sq1_idx + 1, 8):
            if time.time() > deadline - 10:
                break
            sq1 = base_primes[sq1_idx]
            sq2 = base_primes[sq2_idx]
            
            for start in range(0, 3):
                if time.time() > deadline - 10:
                    break
                primes_list = []
                exponents_list = []
                prod = 1
                count = 0
                for i in range(start, start + 15):
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
    
    # Phase 5: Local search on exponent vectors for the best prime set found
    if time.time() > deadline - 5:
        return best_n
    
    # Try to improve by adjusting exponents of the current best
    # Extract prime set from best_n
    best_factors = []
    temp = best_n
    for p in base_primes:
        while temp % p == 0:
            best_factors.append(p)
            temp //= p
        if temp == 1:
            break
        if p * p > temp:
            break
    if temp > 1:
        best_factors.append(temp)
    
    # Convert to (primes, exponents)
    from collections import Counter
    fac_counts = Counter(best_factors)
    bp = [p for p in fac_counts.keys()]
    be = [fac_counts[p] for p in bp]
    bp.sort()
    be = [fac_counts[p] for p in bp]
    
    # Try small perturbations
    for trial in range(50):
        if time.time() > deadline - 3:
            break
        # Try incrementing one exponent
        for i in range(len(bp)):
            if be[i] < 5:
                new_be = be[:]
                new_be[i] += 1
                prod = 1
                for p, e in zip(bp, new_be):
                    prod *= p ** e
                if prod <= 10**18:
                    try_candidate(bp, new_be)
        # Try decrementing one exponent (if > 1)
        for i in range(len(bp)):
            if be[i] > 1:
                new_be = be[:]
                new_be[i] -= 1
                try_candidate(bp, new_be)
        # Try swapping: decrease one, increase another
        for i in range(len(bp)):
            if be[i] > 1:
                for j in range(len(bp)):
                    if i != j and be[j] < 5:
                        new_be = be[:]
                        new_be[i] -= 1
                        new_be[j] += 1
                        prod = 1
                        for p, e in zip(bp, new_be):
                            prod *= p ** e
                        if prod <= 10**18:
                            try_candidate(bp, new_be)
    
    return best_n
