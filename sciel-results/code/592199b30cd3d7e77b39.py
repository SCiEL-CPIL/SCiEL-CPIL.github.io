"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
_start = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def construct():
    # Deadline guard
    if time.time() - _start > _TIMEOUT:
        return [1]
    
    n = 256
    
    # Sieve to find all primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    
    def is_valid(s):
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 256 for x in s):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    def greedy_merge(s):
        """Greedily merge pairs with highest gain"""
        s = list(s)
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_pair = None
            s_sorted = sorted(s)
            for i in range(len(s_sorted)):
                for j in range(i + 1, len(s_sorted)):
                    a, b = s_sorted[i], s_sorted[j]
                    if gcd(a, b) != 1:
                        continue
                    prod = a * b
                    if prod > 256 or prod in s:
                        continue
                    gain = prod - a - b
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (a, b)
            if best_pair:
                a, b = best_pair
                s.remove(a)
                s.remove(b)
                s.append(a * b)
                improved = True
            if time.time() - _start > _TIMEOUT - 5:
                break
        return s
    
    def local_search(s):
        """Local search to find better merges"""
        s = list(s)
        improved = True
        while improved:
            improved = False
            s_sorted = sorted(s)
            for i in range(len(s_sorted)):
                for j in range(i + 1, len(s_sorted)):
                    a, b = s_sorted[i], s_sorted[j]
                    if gcd(a, b) != 1:
                        continue
                    prod = a * b
                    if prod > 256 or prod in s:
                        continue
                    gain = prod - a - b
                    if gain > 0:
                        s.remove(a)
                        s.remove(b)
                        s.append(prod)
                        improved = True
                        break
                if improved:
                    break
            if time.time() - _start > _TIMEOUT - 5:
                break
        return s
    
    candidates = []
    
    # Strategy 1: Start from primes + 1
    r1 = local_search(greedy_merge([1] + primes))
    if is_valid(r1):
        candidates.append(r1)
    
    # Strategy 2: Start from prime powers + 1
    pp = [1]
    for p in primes:
        v = p
        while v * p <= n:
            v *= p
        pp.append(v)
    r2 = local_search(greedy_merge(pp))
    if is_valid(r2):
        candidates.append(r2)
    
    if not candidates:
        return [1] + primes
    
    best = max(candidates, key=sum)
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
