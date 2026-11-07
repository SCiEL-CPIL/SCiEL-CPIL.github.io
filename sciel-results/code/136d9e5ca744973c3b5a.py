"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

# Precompute prime factorization info
MAX_N = 256
# For each number, get its prime factors with exponents
def factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

# Precompute for all numbers 2..256
prime_factors = {}
for n in range(2, MAX_N + 1):
    prime_factors[n] = factorize(n)

# Precompute which primes divide each number
def prime_divisors(n):
    return set(p for p, e in prime_factors[n])

# Precompute the "divisibility pressure": for each a, which pairs (b,c) have a|bc
# This is expensive to precompute fully, so we'll compute on the fly with caching
# Actually, let's precompute for each a, the set of b such that for some c, a|bc
# But we need distinct b,c, so it's more nuanced.

# Let's precompute: for each a, and for each b != a, what values of c make a | b*c?
# This is still large. Let's just use a fast checker.

def is_valid_set(S):
    """Check if set S satisfies: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

# Faster: for each a in S, check if there exist distinct b,c in S\{a} with a|bc
def is_valid_set_fast(S):
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        # Check if there exist distinct b, c in S_list (b != i, c != i, b != c) with a | b*c
        found = False
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            # We need a | b*c for some c != i, c != j
            # a | b*c iff (a/gcd(a,b)) | c
            g = math.gcd(a, b)
            rem = a // g
            if rem == 1:
                # a | b, so a | b*c for any c. Need c != i, c != j.
                # If n >= 3, there exists such c.
                if n >= 3:
                    found = True
                    break
                else:
                    # n < 3, no valid c
                    continue
            else:
                # Need rem | c for some c in S, c != i, c != j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if c % rem == 0:
                        found = True
                        break
                if found:
                    break
        if found:
            return False
    return True

# Even faster with precomputation: for each a, precompute the set of b values
# such that a does not divide b*c for ALL c. But that depends on S.

# Let's use a different approach: precompute for each a, and for each possible b,
# the "required" c values. But this is still O(n^3) in the worst case.

# For n up to ~100, O(n^3) = 1e6 which is fine. Let's just use the fast checker.

def greedy_construct():
    """Greedy: start with primes, try to add composites, remove conflicting primes."""
    primes = [n for n in range(2, MAX_N + 1) if len(prime_factors[n]) == 1 and n != 1]
    # Wait, primes have exactly one prime factor with exponent 1
    primes = [n for n in range(2, MAX_N + 1) if prime_factors[n] and prime_factors[n][0][1] == 1 and len(prime_factors[n]) == 1]
    
    S = set(primes)
    
    # Try to add composites in descending order
    composites = [n for n in range(MAX_N, 1, -1) if n not in S]
    
    improved = True
    while improved:
        improved = False
        for x in composites:
            if x in S:
                continue
            # Try adding x
            S_new = S | {x}
            if is_valid_set_fast(S_new):
                S = S_new
                improved = True
        # Also try swapping: remove some primes to add composites
        for x in composites:
            if x in S:
                continue
            # Find which primes in S would conflict with x
            # Actually, just try removing subsets of primes
            # For now, just try removing one prime at a time
            for p in list(S):
                S_trial = (S - {p}) | {x}
                if is_valid_set_fast(S_trial) and len(S_trial) > len(S):
                    S = S_trial
                    improved = True
                    break
    
    return sorted(S)

def local_search(start_set, max_iters=500, time_limit=60.0):
    """Local search: try swaps and adds."""
    S = set(start_set)
    best = S
    best_size = len(S)
    start_time = time.time()
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit - 2.0:
            break
        current_size = len(S)
        improved = False
        
        # Try adding a number
        for x in range(2, MAX_N + 1):
            if x in S:
                continue
            S_new = S | {x}
            if is_valid_set_fast(S_new) and len(S_new) > current_size:
                S = S_new
                current_size = len(S)
                improved = True
                if current_size > best_size:
                    best = set(S)
                    best_size = current_size
                break
        
        # Try swapping one out, one in
        for x in range(2, MAX_N + 1):
            if x in S:
                continue
            for y in list(S):
                S_new = (S - {y}) | {x}
                if len(S_new) == current_size and is_valid_set_fast(S_new):
                    # Accept with some probability or if it leads to improvement
                    if random.random() < 0.3:
                        S = S_new
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try removing and adding two
            for x in range(2, MAX_N + 1):
                if x in S:
                    continue
                for y in list(S):
                    for z in list(S):
                        if z == y:
                            continue
                        S_new = (S - {y, z}) | {x}
                        if len(S_new) > current_size and is_valid_set_fast(S_new):
                            S = S_new
                            current_size = len(S)
                            improved = True
                            if current_size > best_size:
                                best = set(S)
                                best_size = current_size
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Start with greedy
    S = greedy_construct()
    
    # Local search from greedy result
    best = local_search(S, max_iters=200, time_limit=deadline - start_time - 2.0)
    
    # Multiple restarts with randomization
    while time.time() < deadline - 2.0:
        # Random start: subset of primes
        primes = [n for n in range(2, MAX_N + 1) if len(prime_factors[n]) == 1]
        random.shuffle(primes)
        start_set = set(primes[:random.randint(10, len(primes))])
        candidate = local_search(start_set, max_iters=100, time_limit=10.0)
        if len(candidate) > len(best):
            best = candidate
    
    return sorted(best)
# EVOLVE-BLOCK-END
