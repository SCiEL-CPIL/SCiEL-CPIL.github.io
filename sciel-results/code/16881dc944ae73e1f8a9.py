"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

def _sieve_primes(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, n + 1) if is_prime[i]]

def _prime_factors(num, primes):
    """Return set of prime factors of num."""
    factors = set()
    temp = num
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return factors

def _is_coprime_to_all(a, others, factor_cache):
    """Check if a is coprime to all elements in others using cached factors."""
    fa = factor_cache[a]
    for b in others:
        if fa & factor_cache[b]:  # Intersection of prime factor sets
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill

    n = 256
    primes = _sieve_primes(n)
    
    # Precompute prime factor sets for all numbers
    factor_cache = {}
    for i in range(1, n + 1):
        if i == 1:
            factor_cache[i] = set()
        else:
            factor_cache[i] = _prime_factors(i, primes)

    def time_ok():
        return time.time() - start_time < deadline

    # --- Multiple greedy initializations ---
    candidates = []
    
    # Candidate 1: primes + 1
    prime_set = [1] + primes
    candidates.append((sum(prime_set), list(prime_set)))
    
    # Candidate 2: greedy descending
    selected = [1]
    for a in range(n, 1, -1):
        if not time_ok():
            break
        if _is_coprime_to_all(a, selected, factor_cache):
            selected.append(a)
    candidates.append((sum(selected), list(selected)))
    
    # Candidate 3: greedy by value/number_of_prime_factors (prefer numbers with fewer prime factors)
    selected = [1]
    for a in range(n, 1, -1):
        if not time_ok():
            break
        if len(factor_cache[a]) <= 1:  # Prime or 1
            if _is_coprime_to_all(a, selected, factor_cache):
                selected.append(a)
    # Add composites that fit
    for a in range(n, 1, -1):
        if not time_ok():
            break
        if a not in selected and _is_coprime_to_all(a, selected, factor_cache):
            selected.append(a)
    candidates.append((sum(selected), list(selected)))
    
    # Candidate 4: greedy ascending (sometimes helps)
    selected = [1]
    for a in range(2, n + 1):
        if not time_ok():
            break
        if _is_coprime_to_all(a, selected, factor_cache):
            selected.append(a)
    candidates.append((sum(selected), list(selected)))
    
    # Pick best candidate
    best_sum, best_set = max(candidates, key=lambda x: x[0])
    best_set = list(best_set)
    best_in_set = set(best_set)
    
    # --- Local search with 1-swap moves ---
    max_swap_passes = 20
    for _pass in range(max_swap_passes):
        if not time_ok():
            break
        improved = False
        
        # Try single-element swaps: replace b with larger a
        for b in sorted(best_set, reverse=True):
            if not time_ok():
                break
            if b == 1:
                continue
            others = [x for x in best_set if x != b]
            for a in range(n, b, -1):
                if a in best_in_set:
                    continue
                if _is_coprime_to_all(a, others, factor_cache):
                    # Perform swap
                    best_set.remove(b)
                    best_in_set.discard(b)
                    best_set.append(a)
                    best_in_set.add(a)
                    best_sum += a - b
                    improved = True
                    break
        
        if not improved and time_ok():
            # Try 2-swap moves: replace {b1, b2} with {a1, a2} where a1+a2 > b1+b2
            for i, b1 in enumerate(sorted(best_set, reverse=True)):
                if not time_ok():
                    break
                if b1 == 1:
                    continue
                for j, b2 in enumerate(sorted(best_set, reverse=True)):
                    if not time_ok():
                        break
                    if i == j or b2 == 1 or b2 <= b1:
                        continue
                    
                    others = [x for x in best_set if x != b1 and x != b2]
                    target_sum = b1 + b2
                    
                    # Try to find two numbers a1, a2 that are coprime to each other and to others
                    found = False
                    for a1 in range(n, max(b1, b2), -1):
                        if not time_ok():
                            break
                        if a1 in best_in_set or a1 == b1 or a1 == b2:
                            continue
                        if not _is_coprime_to_all(a1, others, factor_cache):
                            continue
                        for a2 in range(n, a1, -1):
                            if not time_ok():
                                break
                            if a2 in best_in_set or a2 == b1 or a2 == b2:
                                continue
                            if a1 + a2 <= target_sum:
                                continue
                            if gcd(a1, a2) != 1:
                                continue
                            if _is_coprime_to_all(a2, others, factor_cache) and _is_coprime_to_all(a2, [a1], factor_cache):
                                # Perform 2-swap
                                best_set.remove(b1)
                                best_set.remove(b2)
                                best_in_set.discard(b1)
                                best_in_set.discard(b2)
                                best_set.append(a1)
                                best_set.append(a2)
                                best_in_set.add(a1)
                                best_in_set.add(a2)
                                best_sum += a1 + a2 - target_sum
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        
        if not improved:
            break

    # Final polish: try all possible single swaps multiple times
    if time_ok():
        for _ in range(10):
            if not time_ok():
                break
            improved = False
            for b in sorted(best_set, reverse=True):
                if not time_ok():
                    break
                if b == 1:
                    continue
                others = [x for x in best_set if x != b]
                for a in range(n, b, -1):
                    if a in best_in_set:
                        continue
                    if _is_coprime_to_all(a, others, factor_cache):
                        best_set.remove(b)
                        best_in_set.discard(b)
                        best_set.append(a)
                        best_in_set.add(a)
                        best_sum += a - b
                        improved = True
                        break
            if not improved:
                break

    # Ensure return type is list of Python ints
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
