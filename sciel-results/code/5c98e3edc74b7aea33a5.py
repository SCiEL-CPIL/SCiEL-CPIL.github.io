"""Rewritten program: CSP-style backtracking with prime-power basis and memoization."""
import sys
import time
from functools import lru_cache
from itertools import product as iproduct

# Precompute primes up to 128
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

_PRIMES = _primes_up_to(128)
_NUM_PRIMES = len(_PRIMES)

# Map each number in [2,128] to its prime exponent vector (tuple)
# Although we use integer products for the CSP state, this precomputation
# aligns with the "exponent vector distinctness" rationale.
_EXP_VECTORS = {}
for x in range(2, 129):
    v = [0] * _NUM_PRIMES
    n = x
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    _EXP_VECTORS[x] = tuple(v)

def construct():
    sys.setrecursionlimit(10000)
    
    # Time management
    start_time = time.time()
    TIME_LIMIT = 55.0
    
    best_seq = []
    best_len = 0
    
    # 1. Fast Greedy Baseline
    # Pick the smallest valid number to get a good initial lower bound.
    values = []
    products = set()
    for x in range(2, 129):
        if values and x <= values[-1]:
            continue
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
            
    best_len = len(values)
    best_seq = list(values)
    
    # 2. DFS with Memoization and Pruning
    visited = {}
    
    def dfs(last_val, prods_frozen, current_seq):
        nonlocal best_len, best_seq
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Pruning: Can we beat the current best?
        if len(current_seq) + (128 - last_val) <= best_len:
            return
            
        # Update best
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            
        # Memoization check
        key = (last_val, prods_frozen)
        if key in visited:
            if len(current_seq) <= visited[key]:
                return
        else:
            visited[key] = len(current_seq)
            
        # Generate candidates
        for next_val in range(last_val + 1, 129):
            # Calculate new products (suffix products)
            suff = [1]
            p = 1
            for v in reversed(current_seq):
                p *= v
                suff.append(p)
                
            new_prods_list = [next_val * s for s in suff]
            new_prods_set = frozenset(new_prods_list)
            
            # Check for collisions
            if prods_frozen.intersection(new_prods_set):
                continue
                
            new_prods = prods_frozen.union(new_prods_set)
            new_seq = current_seq + [next_val]
            
            # Recursive call
            dfs(next_val, new_prods, new_seq)
            
            # If we found a solution that is optimal, we can stop early?
            # Theoretical max is 127.
            # If best_len == 127, we can break.
            if best_len == 127:
                return

    # Start DFS from empty sequence
    # We can also start from the greedy baseline to save time?
    # If we start from greedy, we might miss better solutions that diverge early.
    # So start from empty.
    dfs(1, frozenset(), [])
    
    return best_seq
