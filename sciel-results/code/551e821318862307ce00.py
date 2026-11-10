"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _sieve_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _prime_factors(x, primes):
    """Return frozenset of prime factors of x."""
    factors = set()
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            factors.add(p)
            while x % p == 0:
                x //= p
    if x > 1:
        factors.add(x)
    return frozenset(factors)

def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill
    
    MAX_N = 256
    primes = _sieve_primes(MAX_N)
    
    # Precompute prime factor sets for all numbers 2..256
    pf = {}
    for x in range(2, MAX_N + 1):
        pf[x] = _prime_factors(x, primes)
    
    # Greedy: process numbers in descending order of value
    # Each "slot" is a set of primes that are "used" by numbers assigned to it
    # A number can be added to a slot if its prime factors are disjoint from the slot's used primes
    # We want to minimize the number of slots (or maximize sum)
    
    # Strategy: sort numbers descending, try to fit each into an existing slot
    # If it fits, add it. If not, create a new slot.
    # Also include 1 (no prime factors, always fits anywhere)
    
    def greedy_solve(order):
        """order: list of numbers to consider (in order)"""
        slots = []  # list of (used_primes_set, [numbers])
        total = 0
        for x in order:
            if x == 1:
                # 1 has no prime factors, can go in any slot or its own
                # Put in first slot or create new
                if slots:
                    slots[0][1].append(x)
                    total += x
                else:
                    slots.append((set(), [x]))
                    total += x
                continue
            px = pf[x]
            placed = False
            for i, (used, nums) in enumerate(slots):
                if not used.intersection(px):
                    # Can add to this slot
                    used.update(px)
                    nums.append(x)
                    total += x
                    placed = True
                    break
            if not placed:
                slots.append((set(px), [x]))
                total += x
        return total, slots
    
    # Try different orderings to find a good solution
    best_sum = 0
    best_result = None
    
    # Ordering 1: descending by value
    order1 = list(range(MAX_N, 1, -1)) + [1]
    s1, _ = greedy_solve(order1)
    if s1 > best_sum:
        best_sum = s1
        best_result = [1] + [x for x in range(MAX_N, 1, -1) if True]  # placeholder
    
    # Let me actually track the result properly
    def greedy_solve_track(order):
        slots = []  # list of (used_primes_set, [numbers])
        for x in order:
            if x == 1:
                if slots:
                    slots[0][1].append(x)
                else:
                    slots.append((set(), [x]))
                continue
            px = pf[x]
            placed = False
            for i, (used, nums) in enumerate(slots):
                if not used.intersection(px):
                    used.update(px)
                    nums.append(x)
                    placed = True
                    break
            if not placed:
                slots.append((set(px), [x]))
        result = []
        for _, nums in slots:
            result.extend(nums)
        return result
    
    # Multiple orderings to try
    orderings = []
    
    # Descending by value
    orderings.append(list(range(MAX_N, 1, -1)) + [1])
    
    # Descending by value, but try to fit larger numbers first
    # Actually, let's try: sort by value descending, that's the natural greedy
    
    # Try: ascending by number of prime factors (simpler numbers first)
    orderings.append(sorted(range(2, MAX_N + 1), key=lambda x: (len(pf[x]), -x)) + [1])
    
    # Try: descending by value, ascending by prime factor count
    orderings.append(sorted(range(2, MAX_N + 1), key=lambda x: (-x, len(pf[x]))) + [1])
    
    # Try: primes first (they're the most flexible), then composites
    orderings.append(primes + sorted([x for x in range(2, MAX_N + 1) if x not in primes], key=lambda x: -x) + [1])
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        result = greedy_solve_track(order)
        s = sum(result)
        if s > best_sum:
            best_sum = s
            best_result = result
    
    # Local improvement: try to swap or move numbers
    # For each number in the result, try removing it and see if we can add a larger number
    # This is complex; let's do a simpler improvement
    
    if best_result is None:
        best_result = [1] + list(range(2, MAX_N + 1))  # fallback, will be invalid
    
    # Ensure 1 is included
    if 1 not in best_result:
        best_result.append(1)
    
    # Verify and return
    # Check all pairs are coprime
    from math import gcd
    valid = True
    for i in range(len(best_result)):
        for j in range(i + 1, len(best_result)):
            if gcd(best_result[i], best_result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to safe solution: 1 and all primes
        best_result = [1] + primes
    
    # Convert to list of Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
