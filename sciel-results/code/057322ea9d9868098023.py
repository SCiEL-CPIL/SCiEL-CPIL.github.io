import time
import random
from math import gcd, isqrt
from itertools import combinations

N = 512
DEADLINE = 85.0

def lcm_ok(x, y):
    return (x // gcd(x, y)) * y <= N

def is_compatible_with_set(x, S):
    for y in S:
        if (x // gcd(x, y)) * y > N:
            return False
    return True

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Strategy: Start with a good seed set and use greedy + local search
    
    # The seed: all numbers 1..22 (since lcm of any two <= 22*21/ gcd... wait)
    # Actually, if all numbers divide LCM, then pairwise lcm divides LCM.
    # We want a set S such that lcm(S) <= 512? No, the condition is pairwise lcm <= 512.
    # If we take all divisors of a number M <= 512, then for any a,b | M, lcm(a,b) | M <= 512.
    # So the set of all divisors of M is a valid set.
    # To maximize cardinality, we want M with many divisors.
    
    # Let's find the number <= 512 with the most divisors.
    best_div_count = 0
    best_M = 1
    best_divisors = []
    
    for M in range(1, N + 1):
        count = 0
        divisors = []
        for i in range(1, isqrt(M) + 1):
            if M % i == 0:
                divisors.append(i)
                if i != M // i:
                    divisors.append(M // i)
        count = len(divisors)
        if count > best_div_count:
            best_div_count = count
            best_M = M
            best_divisors = divisors[:]
    
    # Start with divisors of best_M
    S = set(best_divisors)
    
    # Now try to add more elements greedily
    # An element x can be added if lcm(x, y) <= 512 for all y in S.
    # This is equivalent to: for all y in S, x * y / gcd(x, y) <= 512.
    
    # Precompute a helper: for a candidate x, what is the max y in S such that lcm(x,y) > 512?
    # Actually, just check compatibility.
    
    # Sort candidates 1..512 not in S by "difficulty" or just iterate
    candidates = [x for x in range(1, N + 1) if x not in S]
    
    # Greedy addition
    changed = True
    while changed:
        changed = False
        # Try to add one at a time, in order of x
        # Maybe sort candidates by something? Smaller x tends to be more compatible.
        for x in sorted(candidates):
            if time.time() - start_time > DEADLINE - 5:
                return list(S)
            if is_compatible_with_set(x, S):
                S.add(x)
                candidates.remove(x)
                changed = True
    
    # Local search: try to swap elements to free up space?
    # Or just try to add remaining candidates if we removed some?
    # The greedy above adds as many as possible in a single pass, but the order matters.
    # Let's do multiple random restarts or shuffles.
    
    best_S = set(S)
    best_size = len(S)
    
    # Try several random orderings
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        S_current = set(best_divisors) # Reset to base
        candidates = [x for x in range(1, N + 1) if x not in S_current]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > DEADLINE - 5:
                break
            if is_compatible_with_set(x, S_current):
                S_current.add(x)
        
        if len(S_current) > best_size:
            best_size = len(S_current)
            best_S = set(S_current)
    
    # Try a different base: divisors of other highly composite numbers?
    # Or start with empty set and greedy?
    
    # Let's also try starting with 1..k for various k
    for k in range(1, 23):
        if time.time() - start_time > DEADLINE - 5:
            break
        S_current = set(range(1, k + 1))
        candidates = [x for x in range(1, N + 1) if x not in S_current]
        random.shuffle(candidates)
        for x in candidates:
            if is_compatible_with_set(x, S_current):
                S_current.add(x)
        if len(S_current) > best_size:
            best_size = len(S_current)
            best_S = set(S_current)
    
    # Final local search on best_S: try to add any remaining numbers
    remaining = [x for x in range(1, N + 1) if x not in best_S]
    for x in remaining:
        if time.time() - start_time > DEADLINE - 2:
            break
        if is_compatible_with_set(x, best_S):
            best_S.add(x)
    
    return list(best_S)
