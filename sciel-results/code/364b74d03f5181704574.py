import time
from math import gcd

# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

# Precomputation
MAX_VAL = 256
primes = []
is_prime = [True] * (MAX_VAL + 1)
is_prime[0] = is_prime[1] = False
for i in range(2, MAX_VAL + 1):
    if is_prime[i]:
        primes.append(i)
        for j in range(i*i, MAX_VAL + 1, i):
            is_prime[j] = False

# Map prime to index for bitmask
prime_to_idx = {p: i for i, p in enumerate(primes)}
num_primes = len(primes)

# Precompute mask for each number 1..256
# mask[n] is a bitmask of primes dividing n
masks = [0] * (MAX_VAL + 1)
for n in range(2, MAX_VAL + 1):
    m = 0
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            m |= (1 << prime_to_idx[p])
            while temp % p == 0:
                temp //= p
    if temp > 1:
        m |= (1 << prime_to_idx[temp])
    masks[n] = m

# Precompute multiples for each prime
# multiples[p_idx] = list of n in 1..256 such that p | n, sorted descending
multiples = [[] for _ in range(num_primes)]
for i, p in enumerate(primes):
    mults = []
    for n in range(MAX_VAL, 1, -1):
        if n % p == 0:
            mults.append(n)
    multiples[i] = mults

start_time = time.time()
best_sum = 0
best_set = []

# Sort primes in descending order to handle "large" primes first?
# Or ascending?
# Descending allows tighter bounds earlier for large primes which have fewer options.
sorted_prime_indices = list(range(num_primes - 1, -1, -1))

def get_max_value_for_prime(p_idx, used_mask):
    """Find max n <= 256 divisible by p such that gcd(n, used_primes)=1."""
    p = primes[p_idx]
    # multiples[p_idx] is sorted descending
    for n in multiples[p_idx]:
        if masks[n] & used_mask == 0:
            return n
    return 0

def dfs(idx, used_mask, current_sum, current_set):
    global best_sum, best_set, start_time
    
    # Time check
    if time.time() - start_time > 50:
        return

    # Update best
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = current_set[:]

    # If no more primes, return
    if idx == len(sorted_prime_indices):
        return

    p_idx = sorted_prime_indices[idx]
    p = primes[p_idx]
    
    # Calculate Upper Bound
    # Sum of max possible values for all remaining unused primes
    ub = current_sum
    for next_idx in range(idx + 1, len(sorted_prime_indices)):
        next_p_idx = sorted_prime_indices[next_idx]
        if used_mask & (1 << next_p_idx):
            continue
        # Find max n for this prime given current used_mask
        # Note: used_mask might change in deeper recursion, but for bound
        # we assume current used_mask. This is a valid upper bound.
        val = get_max_value_for_prime(next_p_idx, used_mask)
        ub += val
        
    if ub <= best_sum:
        return

    # Branch 1: Skip p
    dfs(idx + 1, used_mask, current_sum, current_set)
    
    # Branch 2: Use p
    # Find all valid n for p
    # We iterate through multiples of p in descending order
    # To prune, we can stop if current_sum + n + (bound for rest) <= best_sum
    # But simple iteration is fine given small branching factor for large p
    
    for n in multiples[p_idx]:
        if masks[n] & used_mask != 0:
            continue
        
        new_sum = current_sum + n
        if new_sum <= best_sum:
            # Even if we add max possible for rest, can we beat best?
            # Quick check: if new_sum + (sum of all remaining primes values) <= best_sum, skip
            # But we already did a tighter bound at the start of this node.
            # However, choosing a smaller n might allow better combinations for others?
            # No, n consumes primes. Smaller n consumes fewer/smaller primes?
            # Actually, any n consumes the primes in masks[n].
            # If we pick a larger n, we might consume a prime that could have been used for a very large number?
            # But the bound calculation accounts for the *current* used_mask.
            # If we pick n, the new used_mask includes masks[n].
            # The bound for the *next* node will be tighter.
            # So we just recurse.
            pass
        
        new_mask = used_mask | masks[n]
        current_set.append(n)
        dfs(idx + 1, new_mask, new_sum, current_set)
        current_set.pop()

# Start DFS
dfs(0, 0, 0, [])

# Add 1 if not already present (1 is coprime to all)
# 1 has mask 0.
# In our DFS, we only considered primes.
# 1 can always be added.
if 1 not in best_set:
    best_sum += 1
    best_set.append(1)

def construct():
    return list(best_set)

# EVOLVE-BLOCK-END
