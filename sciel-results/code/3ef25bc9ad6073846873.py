"""Improved program: greedy + local search for large equal-gcd-triple-free sets."""
import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0  # safety margin under 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_valid(selected):
    """Check if set has no bad triple."""
    n = len(selected)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = selected[i], selected[j], selected[k]
                if gcd(a, b) == gcd(a, c) == gcd(b, c):
                    return False
    return True

def _can_add(selected, x):
    """Check if adding x keeps the set valid."""
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = selected[i], selected[j]
            if gcd(a, b) == gcd(a, x) == gcd(b, x):
                return False
    return True

def _greedy_construct(order):
    """Build a set greedily following the given order."""
    selected = []
    sel_set = set()
    for x in order:
        if x in sel_set:
            continue
        if _can_add(selected, x):
            selected.append(x)
            sel_set.add(x)
    return selected

def _local_search(selected, max_iters=200):
    """Try to improve by removing elements that block many additions."""
    selected = list(selected)
    sel_set = set(selected)
    all_nums = set(range(1, 129))
    
    for _ in range(max_iters):
        if _time_left() < 10.0:
            break
        
        # For each element, compute how many additions it blocks
        blocked = {}
        for x in selected:
            blocked[x] = 0
        
        # For each candidate not in set, find which elements block it
        candidates = all_nums - sel_set
        for cand in candidates:
            blockers = set()
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    a, b = selected[i], selected[j]
                    if gcd(a, b) == gcd(a, cand) == gcd(b, cand):
                        blockers.add(a)
                        blockers.add(b)
            for b in blockers:
                if b in blocked:
                    blocked[b] += 1
        
        # Find the element blocking the most
        if not blocked or max(blocked.values()) == 0:
            break
        
        worst = max(blocked, key=lambda x: blocked[x])
        
        # Remove it and see if we can add more
        new_selected = [x for x in selected if x != worst]
        new_set = set(new_selected)
        
        # Try to add candidates
        added = 0
        for cand in sorted(candidates, key=lambda x: -x):
            if cand in new_set:
                continue
            if _can_add(new_selected, cand):
                new_selected.append(cand)
                new_set.add(cand)
                added += 1
        
        if len(new_selected) > len(selected):
            selected = new_selected
            sel_set = new_set
    
    return selected

def construct():
    random.seed(42)
    
    # Strategy 1: Start with power-of-2 core, then greedily add
    core = [2 ** e for e in range(0, 8)]
    
    # Multiple greedy attempts with different orderings
    best = list(core)
    
    # Ordering 1: by bit length descending, then by number of shared factors
    def _shared_count(x, s):
        return sum(1 for v in s if gcd(x, v) > 1)
    
    all_nums = list(range(1, 129))
    
    # Try several orderings
    orderings = []
    
    # Ordering A: descending by value
    orderings.append(sorted(all_nums, reverse=True))
    
    # Ordering B: ascending by value  
    orderings.append(sorted(all_nums))
    
    # Ordering C: by number of divisors (more composite first)
    def _num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count
    orderings.append(sorted(all_nums, key=lambda x: -_num_divisors(x)))
    
    # Ordering D: random
    for _ in range(5):
        r = list(all_nums)
        random.shuffle(r)
        orderings.append(r)
    
    # Ordering E: prime powers first
    def _is_prime_power(n):
        if n <= 1:
            return 0
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]:
            if n == p:
                return 1
            pp = p * p
            while pp < n:
                pp *= p
            if pp == n:
                return 2
        return 0
    orderings.append(sorted(all_nums, key=lambda x: (-_is_prime_power(x), -x)))
    
    for order in orderings:
        if _time_left() < 15.0:
            break
        result = _greedy_construct(order)
        if len(result) > len(best):
            best = result
    
    # Local search on best
    if _time_left() > 20.0:
        improved = _local_search(best)
        if len(improved) > len(best):
            best = improved
    
    # Final verification
    if not _is_valid(best):
        # Fallback to core
        best = list(core)
    
    if not best:
        best = [1]
    
    return best

# EVOLVE-BLOCK-END
