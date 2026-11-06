"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Search over exponent vectors for products of small primes.
The key insight is that higher powers of small primes combined with
fewer distinct primes can create more divisors with different adjacency
patterns, potentially yielding more coprime adjacent pairs than the
square-free primorial.
"""

# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product
from collections import defaultdict
import heapq

def _tau_perp_fast(n):
    """Faster tau_perp using factorization-based divisor generation."""
    if n < 2:
        return 0
    # Factorize n
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    
    # Generate divisors via recursive product
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        pw = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _tau_perp_from_factors(factors):
    """Compute tau_perp directly from prime factorization dict {p: e}."""
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        pw = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _estimate_tau_perp(primes, exps):
    """Quick estimate of tau_perp without full divisor generation.
    Based on the observation that coprime adjacent pairs occur when
    consecutive divisors in sorted order have disjoint prime supports.
    """
    total_divs = 1
    for e in exps:
        total_divs *= (e + 1)
    
    # Heuristic: the number of coprime adjacent pairs is roughly
    # proportional to the number of "transitions" between different
    # prime support sets. For square-free numbers with k primes,
    # there are many such transitions.
    k = len([e for e in exps if e > 0])
    
    # Empirical heuristic based on observed patterns
    # More distinct primes with lower exponents tend to give more coprime pairs
    base = k * (k - 1) // 2  # pairs of primes
    # Adjust for exponents: higher exponents create more divisors but
    # also create more "clusters" that reduce coprime adjacency
    exp_penalty = sum(e for e in exps if e > 1)
    estimate = base * max(1, total_divs // 10) - exp_penalty * 5
    
    return max(0, estimate)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # Start with primorial of 15 primes
    P15 = 1
    for p in primes_all[:15]:
        P15 *= p
    
    best_n = P15
    best_score = _tau_perp_fast(P15)
    
    # Also try some other base candidates
    base_candidates = [P15]
    
    # Try 14 primes
    P14 = 1
    for p in primes_all[:14]:
        P14 *= p
    base_candidates.append(P14)
    
    # Try 13 primes with extra exponent on 2
    P13 = 1
    for p in primes_all[:13]:
        P13 *= p
    base_candidates.append(P13)
    base_candidates.append(P13 * 2)
    base_candidates.append(P13 * 4)
    
    # Try 12 primes with extra exponents
    P12 = 1
    for p in primes_all[:12]:
        P12 *= p
    base_candidates.append(P12)
    base_candidates.append(P12 * 4)
    base_candidates.append(P12 * 8)
    base_candidates.append(P12 * 9)
    base_candidates.append(P12 * 12)
    
    # Evaluate all base candidates
    for n_try in base_candidates:
        if n_try <= 10**18 and n_try >= 2:
            s = _tau_perp_fast(n_try)
            if s > best_score:
                best_score = s
                best_n = n_try
    
    # Systematic search over exponent vectors using best-first search
    # Focus on the most promising regions
    
    for k in range(10, 16):
        if time.time() > deadline - 15:
            break
        
        pk = primes_all[:k]
        
        # Start with all exponents = 1 (square-free)
        base_exps = [1] * k
        base_val = 1
        for p, e in zip(pk, base_exps):
            base_val *= p ** e
        
        if base_val > 10**18:
            continue
        
        # Precompute max exponent for each prime position
        max_exps = []
        for i in range(k):
            max_e = 1
            while base_val * (pk[i] ** max_e) <= 10**18:
                max_e += 1
            max_exps.append(max_e)
        
        # Use a priority queue with a better heuristic
        # Score candidates by estimated tau_perp and potential for improvement
        visited = set()
        visited.add(tuple(base_exps))
        
        # Priority queue: (-estimated_score, -total_exponents, exps_tuple, value)
        # We want to explore high-scoring candidates first
        pq = []
        est_score = _estimate_tau_perp(pk, base_exps)
        heapq.heappush(pq, (-est_score, -sum(base_exps), tuple(base_exps), base_val))
        
        evaluated = set()
        max_iterations = 2000
        
        for step in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            if not pq:
                break
            
            neg_est, neg_total, exps_key, val_c = heapq.heappop(pq)
            
            if exps_key in evaluated:
                continue
            evaluated.add(exps_key)
            
            exps_c = list(exps_key)
            
            # Evaluate this candidate
            n_c = 1
            for p, e in zip(pk, exps_c):
                n_c *= p ** e
            
            if n_c <= 10**18 and n_c >= 2:
                s = _tau_perp_fast(n_c)
                if s > best_score:
                    best_score = s
                    best_n = n_c
            
            # Generate neighbors: increment each prime's exponent by 1
            for i in range(k):
                if exps_c[i] + 1 < max_exps[i]:
                    new_exps = exps_c[:]
                    new_exps[i] += 1
                    new_val = val_c * pk[i]
                    if new_val <= 10**18:
                        new_key = tuple(new_exps)
                        if new_key not in visited and new_key not in evaluated:
                            visited.add(new_key)
                            new_est = _estimate_tau_perp(pk, new_exps)
                            heapq.heappush(pq, (-new_est, -sum(new_exps), new_key, new_val))
            
            # Also try: decrement one and increment another (swap)
            for i in range(k):
                if exps_c[i] > 1:
                    for j in range(k):
                        if i != j and exps_c[j] + 1 < max_exps[j]:
                            new_exps = exps_c[:]
                            new_exps[i] -= 1
                            new_exps[j] += 1
                            new_val = val_c * (pk[j] // pk[i])
                            if new_val <= 10**18:
                                new_key = tuple(new_exps)
                                if new_key not in visited and new_key not in evaluated:
                                    visited.add(new_key)
                                    new_est = _estimate_tau_perp(pk, new_exps)
                                    heapq.heappush(pq, (-new_est, -sum(new_exps), new_key, new_val))
    
    # Additional targeted searches for specific patterns
    # Pattern: 2^a * 3^b * 5^c * 7^d * product of remaining primes
    if time.time() < deadline - 10:
        for k in range(8, 16):
            pk = primes_all[:k]
            base_val = 1
            for p in pk:
                base_val *= p
            
            if base_val > 10**18:
                continue
            
            # Extra budget for 2, 3, 5, 7
            budget = 10**18 // base_val
            
            for a in range(0, min(12, budget // 2 + 1)):
                rem1 = budget // (2 ** a) if a > 0 else budget
                if rem1 < 1:
                    break
                for b in range(0, min(8, rem1 // 3 + 1)):
                    rem2 = rem1 // (3 ** b) if b > 0 else rem1
                    if rem2 < 1:
                        break
                    for c in range(0, min(6, rem2 // 5 + 1)):
                        rem3 = rem2 // (5 ** c) if c > 0 else rem2
                        if rem3 < 1:
                            break
                        for d in range(0, min(4, rem3 // 7 + 1)):
                            extra = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                            n_try = base_val * extra
                            if n_try <= 10**18 and n_try >= 2:
                                if time.time() > deadline - 5:
                                    break
                                s = _tau_perp_fast(n_try)
                                if s > best_score:
                                    best_score = s
                                    best_n = n_try
    
    # Final local optimization: try small perturbations around the best found
    if time.time() < deadline - 3:
        best_factors = {}
        temp = best_n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                best_factors[d] = best_factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            best_factors[temp] = best_factors.get(temp, 0) + 1
        
        best_primes = list(best_factors.keys())
        best_exps = [best_factors[p] for p in best_primes]
        
        # Try incrementing/decrementing each exponent by 1
        for i in range(len(best_primes)):
            for delta in [-1, 1]:
                if best_exps[i] + delta >= 1:
                    new_exps = best_exps[:]
                    new_exps[i] += delta
                    n_try = 1
                    for p, e in zip(best_primes, new_exps):
                        n_try *= p ** e
                    if 2 <= n_try <= 10**18:
                        s = _tau_perp_fast(n_try)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
    
    return best_n
# EVOLVE-BLOCK-END
