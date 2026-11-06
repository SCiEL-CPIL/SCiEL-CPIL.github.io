import sys
from itertools import combinations

# Precompute smooth numbers up to 128
SMOOTH_NUMS = []
for i in range(1, 129):
    x = i
    while x % 2 == 0: x //= 2
    while x % 3 == 0: x //= 3
    while x % 5 == 0: x //= 5
    if x == 1:
        SMOOTH_NUMS.append(i)

def is_antichain(terms):
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            a, b = terms[i], terms[j]
            if a % b == 0 or b % a == 0:
                return False
    return True

def find_min_antichain(n):
    # Try increasing sizes
    for k in range(1, 11): # Max 30, but usually much smaller
        # Filter smooth numbers <= n
        candidates = [x for x in SMOOTH_NUMS if x <= n]
        # If k is 1
        if k == 1:
            if n in candidates:
                return [n]
        elif k == 2:
            # Optimization: iterate pairs
            for i in range(len(candidates)):
                a = candidates[i]
                b = n - a
                if b in candidates and b != a and b > a: # Avoid duplicates and self
                    if a % b != 0 and b % a != 0:
                        return [a, b]
        else:
            # General case for k >= 3
            # Use combinations, but prune heavily
            # We can use a recursive helper or just combinations for small k
            # For k=3, combinations of 3 is manageable
            # For larger k, we might need more pruning, but N=128 is small
            # Let's just use combinations for now, it's fast enough for N<=128
            found = None
            for comb in combinations(candidates, k):
                if sum(comb) == n and is_antichain(comb):
                    found = list(comb)
                    break
            if found:
                return found
    return None # Should not happen

# Pre-compute all answers
_PRECOMPUTED = []
for n in range(1, 129):
    _PRECOMPUTED.append(find_min_antichain(n))

def construct():
    return _PRECOMPUTED
