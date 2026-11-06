import time
import math

# Global state for timing
START_TIME = time.time()
TIME_LIMIT = 50.0

def _sieve_spf(limit):
    spf = list(range(limit + 1))
    for i in range(4, limit + 1, 2):
        spf[i] = 2
    for i in range(3, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Precompute SPF up to 20000 (sums of two numbers <= 10000 are <= 20000)
SPF_LIMIT = 20000
SPF = _sieve_spf(SPF_LIMIT)

def _get_primes(n):
    primes = set()
    while n > 1:
        p = SPF[n]
        primes.add(p)
        while n % p == 0:
            n //= p
    return primes

def _is_smooth(n, allowed_primes):
    if n <= 1:
        return True
    # Quick check: if n is in allowed primes, return True
    # But n is a sum, so n >= 3.
    # Factorize
    curr = n
    while curr > 1:
        p = SPF[curr]
        if p not in allowed_primes:
            return False
        while curr % p == 0:
            curr //= p
    return True

def _search_set(n, allowed_primes, max_val):
    # Greedy with backtracking
    # We want to find n distinct integers from 1..max_val
    # such that all pairwise sums are smooth wrt allowed_primes
    
    # To make search efficient, we can pre-filter candidates?
    # A candidate x is only useful if x+y is smooth for some y.
    # But we don't know y yet.
    
    # Simple backtracking
    current_set = []
    
    def backtrack(start_idx, current_set):
        if len(current_set) == n:
            return current_set[:]
        
        if time.time() - START_TIME > TIME_LIMIT:
            return None
            
        # Try candidates from start_idx to max_val
        # Optimization: We can skip candidates that are too large?
        # Or just iterate.
        
        for i in range(start_idx, max_val + 1):
            candidate = i
            # Check compatibility with all in current_set
            compatible = True
            for x in current_set:
                if not _is_smooth(x + candidate, allowed_primes):
                    compatible = False
                    break
            
            if compatible:
                res = backtrack(i + 1, current_set + [candidate])
                if res is not None:
                    return res
        return None

    return backtrack(1, [])

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n = 12
    max_val = 10000
    
    # Try prime sets in order of increasing size
    # We want to minimize the number of distinct primes
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 13, 17},
        {2, 3, 5, 7, 11, 13, 17, 19},
    ]
    
    best_result = None
    best_score = float('inf')
    
    for primes in prime_sets_to_try:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        result = _search_set(n, primes, max_val)
        if result:
            # Verify result
            valid = True
            union_primes = set()
            for i in range(n):
                for j in range(i + 1, n):
                    s = result[i] + result[j]
                    p_set = _get_primes(s)
                    union_primes.update(p_set)
            
            # The score is the number of distinct primes
            score = len(union_primes)
            
            if score < best_score:
                best_score = score
                best_result = result
                
            # If we found a set with primes exactly matching the target set, 
            # and the target set size is small, we can stop early if we are confident.
            # But we want the MINIMUM number of primes.
            # If we are trying {2,3}, and we found a set, the primes are subset of {2,3}.
            # The actual primes might be just {2} or {3} or {2,3}.
            # We can break if we find a set with score 1 or 2, as it's hard to beat.
            if score <= 2:
                return result
                
    if best_result:
        return best_result
        
    # Fallback: return seed or a simple set
    return list(range(1, 13))
