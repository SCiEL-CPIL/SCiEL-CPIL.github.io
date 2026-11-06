# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85.0  # 85s safety margin (limit is 90s)
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute lcm for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    def has_bad_triple(selected, x):
        """Check if adding x to selected creates a bad triple."""
        n = len(selected)
        # For each pair (a,b) in selected, check if lcm(a,b)==lcm(a,x)==lcm(b,x)
        # Equivalently: lcm(a,b) == lcm(a,x) and lcm(a,b) == lcm(b,x)
        # which means lcm(a,b) divides lcm(a,x) and lcm(a,x) divides lcm(a,b)
        # i.e., lcm(a,b) == lcm(a,x) == lcm(b,x)
        for i in range(n):
            a = selected[i]
            lax = get_lcm(a, x)
            for j in range(i + 1, n):
                b = selected[j]
                if get_lcm(a, b) == lax and get_lcm(b, x) == lax:
                    return True
        return False
    
    def greedy(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if not has_bad_triple(selected, x):
                selected.append(x)
        return selected
    
    # Try multiple orderings
    best = []
    
    # Ordering 1: ascending
    s1 = greedy(nums)
    if len(s1) > len(best):
        best = s1
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 2: descending
    s2 = greedy(nums[::-1])
    if len(s2) > len(best):
        best = s2
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 3: powers of 2 first, then odd, then even non-powers
    def sort_key(x):
        # Priority: powers of 2 first (they're safe with each other), then by some heuristic
        # Actually let's try: sort by number of divisors descending (more "connected" first)
        # Or: sort by value
        return x
    s3 = greedy(sorted(nums, key=lambda x: (x & 1, x)))  # odd first, then even
    if len(s3) > len(best):
        best = s3
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Ordering 4: by number of prime factors (more factors = more constraints, add later)
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    s4 = greedy(sorted(nums, key=omega))
    if len(s4) > len(best):
        best = s4
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Try to improve: local search - try swapping elements
    # For each element not in best, try to add it and remove one that conflicts
    improved = list(best)
    remaining = set(nums) - set(improved)
    
    # Try adding each remaining element one at a time, removing minimal conflicts
    for x in list(remaining):
        if time.time() > deadline:
            break
        # Check if x can be added
        if not has_bad_triple(improved, x):
            improved.append(x)
            remaining.discard(x)
    
    if len(improved) > len(best):
        best = improved
    
    return best if best else [1]
# EVOLVE-BLOCK-END
