"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 256
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Base sum: all primes + 1
    # We will try to improve the sum by replacing subsets of primes with their product
    # if the product <= N and product > sum of the subset.
    
    # Identify active primes (those that can be part of a merge)
    # A prime p can be part of a merge if there exists q != p such that p*q <= N.
    # This implies p <= N/2 = 128.
    active_primes = [p for p in primes if p <= N // 2]
    inactive_primes = [p for p in primes if p > N // 2]
    
    # Generate all potential merge groups
    # A group is a tuple of primes.
    # Valid if product <= N and product > sum(primes in group).
    # Groups must have size >= 2.
    
    # We'll store groups as (frozenset, gain, product)
    # gain = product - sum(primes)
    
    potential_groups = []
    
    # Helper to check if time is up
    def time_check():
        return time.time() > deadline

    # Check pairs
    for i in range(len(active_primes)):
        if time_check():
            break
        p1 = active_primes[i]
        for j in range(i + 1, len(active_primes)):
            p2 = active_primes[j]
            prod = p1 * p2
            if prod > N:
                break
            s = p1 + p2
            if prod > s:
                potential_groups.append((frozenset([p1, p2]), prod - s, prod))

    # Check triples
    for i in range(len(active_primes)):
        if time_check():
            break
        p1 = active_primes[i]
        for j in range(i + 1, len(active_primes)):
            p2 = active_primes[j]
            prod12 = p1 * p2
            if prod12 > N // 2: # Since min p3 is 2, if prod12 > N/2, prod12*p3 > N for all p3 >= 2? No, p3 could be 2.
                 # Actually just check next
                pass
            for k in range(j + 1, len(active_primes)):
                p3 = active_primes[k]
                prod = prod12 * p3
                if prod > N:
                    break
                s = p1 + p2 + p3
                if prod > s:
                    potential_groups.append((frozenset([p1, p2, p3]), prod - s, prod))

    # Check quads
    # Only possible with very small primes.
    # 2*3*5*7 = 210.
    # 2*3*5*11 = 330 > 256.
    # So only one quad to check?
    # Let's just generate quads from first few active primes.
    small_active = active_primes[:10]
    for combo in combinations(small_active, 4):
        prod = 1
        for p in combo:
            prod *= p
        if prod > N:
            continue
        s = sum(combo)
        if prod > s:
            potential_groups.append((frozenset(combo), prod - s, prod))
            
    # Sort groups by gain descending to prioritize high-value merges in search
    potential_groups.sort(key=lambda x: x[1], reverse=True)
    
    best_gain = 0
    best_groups = []
    
    # Backtracking search to find max weight independent set
    # State: index in potential_groups, current set of used primes, current total gain
    # Since the number of groups is small (likely < 100), and active primes < 30,
    # we can do DFS.
    
    active_prime_set = set(active_primes)
    
    # Memoization: (idx, used_primes) -> best gain achieved from this state
    memo = {}
    _time_check_counter = [0]
    
    def backtrack(idx, used_primes, current_gain, current_groups):
        nonlocal best_gain, best_groups
        
        # Check time every 1000 calls
        _time_check_counter[0] += 1
        if _time_check_counter[0] % 1000 == 0 and time.time() > deadline:
            return
        
        if current_gain > best_gain:
            best_gain = current_gain
            best_groups = list(current_groups)
            
        if idx == len(potential_groups):
            return
            
        # Memoization check
        state_key = (idx, used_primes)
        if state_key in memo:
            if memo[state_key] >= current_gain:
                return
        else:
            memo[state_key] = current_gain
            
        # Option 1: Skip this group
        backtrack(idx + 1, used_primes, current_gain, current_groups)
        
        # Option 2: Take this group if it doesn't conflict
        group_set, gain, prod = potential_groups[idx]
        if not used_primes.isdisjoint(group_set):
            return
        backtrack(idx + 1, used_primes | group_set, current_gain + gain, current_groups + [group_set])

    backtrack(0, frozenset(), 0, [])
    
    # Construct the final set
    # Start with all primes
    final_set = set(primes)
    final_set.add(1)
    
    # Apply the best groups found
    # For each group in best_groups, we remove the primes in the group and add the product
    for group in best_groups:
        prod = 1
        for p in group:
            final_set.remove(p)
            prod *= p
        final_set.add(prod)
        
    result = list(final_set)
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
