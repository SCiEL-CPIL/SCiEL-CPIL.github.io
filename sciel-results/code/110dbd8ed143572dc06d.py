"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 256
    
    # 1. Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Base sum is 1 + sum of all prime powers
    # base_sum = 1 + sum(prime_powers.values())
    
    # 3. Generate potential merge groups
    # Active primes for merging are p <= N // 2
    active_primes = [p for p in primes if p <= N // 2]
    
    # Map active primes to indices for bitmask representation
    prime_to_idx = {p: i for i, p in enumerate(active_primes)}
    n_active = len(active_primes)
    
    potential_groups = []
    
    def time_check():
        return time.time() > deadline

    # Generate pairs
    for i in range(n_active):
        if time_check():
            break
        p1 = active_primes[i]
        idx1 = i
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod = p1 * p2
            if prod > N:
                break
            cost = prime_powers[p1] + prime_powers[p2]
            gain = prod - cost
            if gain > 0:
                mask = (1 << idx1) | (1 << j)
                potential_groups.append((mask, gain, prod, [p1, p2]))

    # Generate triples
    for i in range(n_active):
        if time_check():
            break
        p1 = active_primes[i]
        idx1 = i
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod12 = p1 * p2
            if prod12 * 2 > N:
                continue
            idx2 = j
            for k in range(j + 1, n_active):
                p3 = active_primes[k]
                prod = prod12 * p3
                if prod > N:
                    break
                cost = prime_powers[p1] + prime_powers[p2] + prime_powers[p3]
                gain = prod - cost
                if gain > 0:
                    mask = (1 << idx1) | (1 << idx2) | (1 << k)
                    potential_groups.append((mask, gain, prod, [p1, p2, p3]))

    # Generate quads
    small_active = active_primes[:10]
    for combo in combinations(small_active, 4):
        prod = 1
        for p in combo:
            prod *= p
        if prod > N:
            continue
        cost = sum(prime_powers[p] for p in combo)
        gain = prod - cost
        if gain > 0:
            mask = 0
            for p in combo:
                mask |= (1 << prime_to_idx[p])
            potential_groups.append((mask, gain, prod, list(combo)))

    # Sort groups by gain descending
    potential_groups.sort(key=lambda x: x[1], reverse=True)
    
    # Precompute suffix sums of gains for pruning
    n_groups = len(potential_groups)
    suffix_gain = [0] * (n_groups + 1)
    for i in range(n_groups - 1, -1, -1):
        suffix_gain[i] = suffix_gain[i + 1] + potential_groups[i][1]

    # 4. Search for max weight independent set of groups
    best_gain = 0
    best_groups = []
    
    def backtrack(idx, used_mask, current_gain, current_groups):
        nonlocal best_gain, best_groups
        
        if time.time() > deadline:
            return
            
        if current_gain > best_gain:
            best_gain = current_gain
            best_groups = list(current_groups)
            
        if idx == n_groups:
            return
            
        # Pruning: even if we take all remaining groups, we can't beat best_gain
        if current_gain + suffix_gain[idx] <= best_gain:
            return
            
        # Option 1: Skip
        backtrack(idx + 1, used_mask, current_gain, current_groups)
        
        # Option 2: Take
        group_mask, gain, prod, primes_in_group = potential_groups[idx]
        if used_mask & group_mask:
            return
        backtrack(idx + 1, used_mask | group_mask, current_gain + gain, current_groups + [group_mask])

    backtrack(0, 0, 0, [])
    
    # 5. Construct final result
    result = [1]
    used_in_merge = set()
    for mask in best_groups:
        # Reconstruct primes from mask
        group_primes = []
        for i in range(n_active):
            if mask & (1 << i):
                p = active_primes[i]
                group_primes.append(p)
                used_in_merge.add(p)
        prod = 1
        for p in group_primes:
            prod *= p
        result.append(prod)
        
    for p in primes:
        if p not in used_in_merge:
            result.append(prime_powers[p])
            
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
