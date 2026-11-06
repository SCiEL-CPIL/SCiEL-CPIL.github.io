"""
Reciprocal dissociated set search.

We want a subset S of [1,22] maximizing |S| such that all subset sums of
{1/s : s in S} are distinct (i.e., the set is dissociated over the rationals).

Key insight: 1/a = 1/b + 1/c + ... can only happen with specific relations.
For small sets, we can check via integer arithmetic using LCM of denominators.

Strategy:
1. Precompute a good candidate using greedy + local search with fast integer checks.
2. Use LCM-based integer arithmetic to avoid slow Fraction operations.
3. Try many random restarts and local improvements within a time budget.
"""

import time
import random
from math import gcd
from functools import reduce
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _is_dissociated_fast(S):
    """
    Check if all subset sums of {1/s for s in S} are distinct.
    Uses integer arithmetic: multiply by LCM of all elements.
    """
    n = len(S)
    if n == 0:
        return True
    if n > 20:
        # Too many to check all 2^n subsets quickly; use modular checks
        return _is_dissociated_modular(S)
    
    # Compute LCM of all elements
    L = reduce(_lcm, S, 1)
    
    # Convert to integers: a_i = L / s_i
    ints = [L // s for s in S]
    
    # Check all subset sums for collisions
    # Use a set of seen sums
    seen = set()
    seen.add(0)
    
    # Iterate over all 2^n subsets
    for mask in range(1, 1 << n):
        s = 0
        # Compute subset sum
        for i in range(n):
            if mask & (1 << i):
                s += ints[i]
        if s in seen:
            return False
        seen.add(s)
    
    return True

def _is_dissociated_modular(S, moduli=(1000003, 1000033, 999983)):
    """
    Modular check: if a collision exists in integers, it likely exists mod p.
    Not perfectly sound but fast for large sets.
    """
    n = len(S)
    for p in moduli:
        # Check if p divides any element (would be problematic)
        valid = True
        for s in S:
            if s % p == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Compute modular inverses
        mods = [pow(s, p - 2, p) for s in S]
        
        # Check subset sums mod p
        seen = set()
        seen.add(0)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s = (s + mods[i]) % p
            if s in seen:
                return False
            seen.add(s)
    return True

def _try_add(S, candidate, L_cache=None):
    """Try adding candidate to S; return True if still dissociated."""
    new_S = S + [candidate]
    return _is_dissociated_fast(new_S)

def _greedy_build():
    """Greedy: add elements in order of smallest reciprocal (largest number) first."""
    S = []
    # Try adding from largest to smallest (larger numbers have smaller reciprocals,
    # making collisions less likely)
    for x in range(22, 0, -1):
        if _try_add(S, x):
            S.append(x)
    return S

def _local_search(S, max_iters=5000):
    """
    Local search: try swapping or adding/removing elements to increase size.
    """
    S = list(S)
    current_size = len(S)
    best = S[:]
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        
        # Try adding a random element not in S
        if len(S) < 22:
            candidates = [x for x in range(1, 23) if x not in S]
            random.shuffle(candidates)
            for c in candidates[:5]:  # Try a few
                if _is_dissociated_fast(S + [c]):
                    S.append(c)
                    improved = True
                    if len(S) > current_size:
                        current_size = len(S)
                        best = S[:]
                    break
        
        # Try swapping: remove one, add another
        if not improved and len(S) >= 2:
            i = random.randint(0, len(S) - 1)
            removed = S[i]
            S_new = S[:i] + S[i+1:]
            candidates = [x for x in range(1, 23) if x not in S_new and x != removed]
            random.shuffle(candidates)
            for c in candidates[:5]:
                if _is_dissociated_fast(S_new + [c]):
                    S = S_new + [c]
                    if len(S) > current_size:
                        current_size = len(S)
                        best = S[:]
                    improved = True
                    break
            if not improved:
                S = S_new + [removed]  # Restore
        
        if not improved:
            # Random perturbation
            if len(S) > 1:
                i = random.randint(0, len(S) - 1)
                removed = S[i]
                S = S[:i] + S[i+1:]
                candidates = [x for x in range(1, 23) if x not in S]
                if candidates:
                    S.append(random.choice(candidates))
    
    return best

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with greedy solution
    best = _greedy_build()
    best_size = len(best)
    
    # Multiple random restarts with local search
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > DEADLINE:
            break
        
        # Random initial set
        if restart == 0:
            S = best[:]
        else:
            # Random subset of size 8-12
            size = rng.randint(8, min(12, 22))
            S = rng.sample(range(1, 23), size)
            # Verify it's valid
            if not _is_dissociated_fast(S):
                # Try to fix: remove conflicting elements
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        if _is_dissociated_fast([S[i], S[j]]):
                            pass
                # Just try greedy from scratch if random fails
                S = _greedy_build()
        
        # Local search
        result = _local_search(S, max_iters=3000)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            
            # If we hit a high size, try to push further
            if best_size >= 12:
                # More aggressive local search
                for extra in range(5):
                    if time.time() > DEADLINE:
                        break
                    result2 = _local_search(best, max_iters=5000)
                    if len(result2) > best_size:
                        best = result2
                        best_size = len(result2)
    
    # Final verification
    if not _is_dissociated_fast(best):
        # Fallback to known good set
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best)

# Pre-compute a good answer at import time if possible
# Actually, let's just make sure construct() is fast enough

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
