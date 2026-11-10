"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit) if is_prime[i]]

PRIMES = _sieve_primes(_N)

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

# Precompute prime masks for each number
prime_to_bit = {p: i for i, p in enumerate(PRIMES)}
_MASKS = {}
for i in range(1, _N + 1):
    pf = _NUM_FACTORS[i]
    mask = 0
    for p in pf:
        mask |= (1 << prime_to_bit[p])
    _MASKS[i] = mask

def _branch_and_bound(deadline):
    """
    Branch-and-bound for maximum weight independent set on conflict graph.
    Nodes: integers 1..256
    Edges: two nodes connected if they share a prime factor
    Weight: the integer value itself
    """
    # Build adjacency list: for each number, which other numbers conflict with it
    # Two numbers conflict if they share a prime factor, i.e., gcd > 1
    # We'll use bitmasks for efficiency
    
    # Numbers to consider: 1..256
    # 1 has no prime factors, so it's coprime to everything
    # For 2..256, we need to check prime factor conflicts
    
    # Sort numbers by value descending for better pruning
    numbers = list(range(_N, 0, -1))
    
    # Precompute conflict masks: for each number, which other numbers conflict
    # Use a dict mapping number -> set of conflicting numbers
    conflict = {n: set() for n in range(1, _N + 1)}
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if _MASKS[i] & _MASKS[j]:  # Share at least one prime factor
                conflict[i].add(j)
                conflict[j].add(i)
    
    # For branch and bound, we'll use a recursive approach with memoization-like pruning
    # State: (current_number_index, used_mask, current_sum)
    # But used_mask can be huge, so we track which numbers are "blocked" by selected numbers
    
    # Better approach: process numbers in order, decide include/exclude
    # Key insight: if we include number n, all numbers that conflict with n are excluded
    # Upper bound: sum of all remaining numbers that don't conflict with already-selected numbers
    
    best_sum = [0]
    best_set = [set()]
    
    def _upper_bound(start_idx, used_mask):
        """
        Compute an upper bound on the sum we can achieve from numbers[start_idx:]
        given that used_mask primes are already taken.
        Simple bound: sum of all numbers from start_idx onward that are coprime to used_mask
        """
        bound = 0
        for idx in range(start_idx, len(numbers)):
            n = numbers[idx]
            if _MASKS[n] & used_mask == 0:
                bound += n
        return bound
    
    def _dfs(idx, used_mask, current_sum, selected):
        """
        Recursive branch and bound.
        idx: index in numbers array
        used_mask: bitmask of primes already used
        current_sum: sum of selected numbers so far
        selected: set of selected numbers
        """
        if time.time() > deadline:
            return
        
        if idx >= len(numbers):
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(selected)
            return
        
        n = numbers[idx]
        
        # Upper bound: if even taking all compatible remaining numbers can't beat best, prune
        ub = current_sum + _upper_bound(idx, used_mask)
        if ub <= best_sum[0]:
            return
        
        # Option 1: Exclude n
        _dfs(idx + 1, used_mask, current_sum, selected)
        
        # Option 2: Include n (if compatible)
        if _MASKS[n] & used_mask == 0:
            new_mask = used_mask | _MASKS[n]
            selected.add(n)
            _dfs(idx + 1, new_mask, current_sum + n, selected)
            selected.remove(n)
    
    _dfs(0, 0, 0, set())
    return best_set[0], best_sum[0]

def construct():
    """
    Find a set of distinct integers from 1..256 such that every pair is coprime,
    maximizing their sum.
    Uses branch-and-bound with upper bound pruning.
    """
    deadline = time.time() + 50  # Safety margin
    
    # Try branch and bound
    selected, total = _branch_and_bound(deadline)
    
    # If B&B didn't find anything or timed out with no result, fall back to greedy
    if not selected:
        # Greedy fallback
        selected = {1}
        used_primes = set()
        for n in range(_N, 1, -1):
            if time.time() > deadline:
                break
            pf = _NUM_FACTORS[n]
            if pf.isdisjoint(used_primes):
                used_primes.update(pf)
                selected.add(n)
        total = sum(selected)
    
    result = sorted(selected)
    
    # Verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to just 1 if something went wrong
                return [1]
                
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
