import time
import random
import math

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

# Precompute candidate sets
# We want to find 12 numbers whose pairwise sums only involve a small set of primes.
# Let's try a construction based on powers of 2 and 3.
# If we pick numbers of the form a * 2^i * 3^j, sums might be complex.
# However, if we pick all numbers to be multiples of a large power of 2, say 2^10=1024,
# then sums are multiples of 1024.
# Let's try a different approach: 
# If we choose 12 numbers that are all congruent to -1 mod some prime p, 
# then a+b is congruent to -2 mod p. This doesn't force p to divide a+b.

# Let's look at the "2-prime" solution.
# If we can make all sums divisible by 2 and 3, we need a+b to be a power of 2 * power of 3.
# This is equivalent to a+b being 2,3-smooth.
# Can we find 12 distinct numbers <= 10000 such that all pairwise sums are 2,3-smooth?
# Let S be the set of 2,3-smooth numbers. We need a_i + a_j in S.
# Let's try a_i = 2^i * 3^j? No.
# What if we take a_i = 2^k_i? 
# a_i + a_j = 2^k_i + 2^k_j = 2^min(k_i, k_j) * (1 + 2^{|k_i-k_j|}).
# For this to be 2,3-smooth, (1 + 2^{|k_i-k_j|}) must be 3-smooth (a power of 3).
# 1 + 2^d = 3^e.
# d=1: 3=3^1 (ok)
# d=2: 5 (no)
# d=3: 9=3^2 (ok)
# d=4: 17 (no)
# d=5: 33 (no)
# d=6: 65 (no)
# d=7: 129 (no)
# d=8: 257 (no)
# d=9: 513 = 3^3 * 19 (no)
# d=10: 1025 (no)
# d=11: 2049 (no)
# d=12: 4097 (no)
# It seems only d=1 and d=3 work for small d.
# So we can have chains of indices where differences are 1 or 3.
# Let's try to construct a set of 12 indices k_1...k_12 such that |k_i - k_j| is either 1 or 3 (or combinations that work).
# Actually, if |k_i - k_j| = 1, sum is 2^min * 3.
# If |k_i - k_j| = 3, sum is 2^min * 9 = 2^min * 3^2.
# Both are 2,3-smooth.
# Can we pack 12 indices such that all differences are 1 or 3?
# No, if we have 0, 1, 4, 5, 8, 9... 
# 0 and 4: diff 4 -> 1+16=17 (no).
# So we can't have 0 and 4.
# We need a graph where edges exist only for diff 1 or 3.
# This graph is sparse. Max clique size?
# 0, 1, 3. 
# 0-1 (1), 1-3 (2? no, 2 is not 1 or 3). 
# So 0, 1, 3 is not a clique because 1 and 3 have diff 2.
# 0, 3, 4? 
# 0-3 (3), 3-4 (1), 0-4 (4, no).
# It seems hard to get 12 numbers with just 2 and 3.

# Let's try adding prime 5.
# Sums divisible by 2, 3, 5.
# This gives more flexibility.
# Let's try to find 12 numbers such that a_i + a_j is 2,3,5-smooth.
# We can do a search.

# Heuristic: 
# Start with a set of 12 random numbers.
# Use local search to minimize the number of distinct primes in the sums.
# To speed up, we can precompute the prime factors of all possible sums? 
# No, sums go up to 20000. We can precompute the set of prime factors for all n in [2, 20000].
# Then count_distinct_primes is just a union of sets.

prime_factors_cache = [set() for _ in range(MAX_VAL + 1)]
for i in range(2, MAX_VAL + 1):
    prime_factors_cache[i] = get_prime_factors(i)

def count_distinct_primes_fast(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(prime_factors_cache[s])
    return len(primes)

# Let's try a specific construction.
# If we take a_i = 2^i, we saw it's limited.
# What if we take a_i = 2^i * 3? 
# a_i + a_j = 3 * (2^i + 2^j). Same issue.

# Let's try a_i = 6 * 2^i.
# Sums are 6 * (2^i + 2^j).
# We need 2^i + 2^j to be 2-smooth? No, it already has a factor of 2.
# 2^i + 2^j = 2^min(i,j) * (1 + 2^{|i-j|}).
# So a_i + a_j = 6 * 2^min(i,j) * (1 + 2^{|i-j|}).
# Primes are {2, 3} union primes of (1 + 2^{|i-j|}).
# So we just need (1 + 2^{|i-j|}) to have few primes.
# We can choose i, j such that |i-j| is small or specific.
# If we choose 12 exponents i_1...i_12 from {0, 1, 2, 3, ...},
# and we want all |i_p - i_q| to be in a set D such that 1+2^d has few primes.
# d=1: 3 (prime 3)
# d=2: 5 (prime 5)
# d=3: 9 (prime 3)
# d=4: 17 (prime 17)
# d=5: 33 (primes 3, 11)
# d=6: 65 (primes 5, 13)
# d=7: 129 (primes 3, 43)
# d=8: 257 (prime 257)
# d=9: 513 (primes 3, 19)
# d=10: 1025 (primes 5, 41)
# d=11: 2049 (primes 3, 683)
# d=12: 4097 (primes 17, 241)
# d=13: 8193 (primes 3, 2731)
# d=14: 16385 (primes 5, 11, 297)
# d=15: 32769 (primes 3, 10923)
# d=16: 65537 (prime 65537)
#
# If we restrict differences to {1, 2, 3}, the primes involved are {2, 3, 5}.
# Can we fit 12 numbers with pairwise differences in {1, 2, 3}?
# No, max clique in a path graph with edges for diff 1,2,3 is 4 (e.g. 0,1,2,3).
# 0 and 4 have diff 4.
# So we can't get 12 numbers with just 2,3,5.

# Let's try differences in {1, 2, 3, 5}.
# d=5 gives prime 11. So primes {2, 3, 5, 11}.
# Max clique size for differences in {1,2,3,5}?
# This is a graph where nodes are integers and edges connect i,j if |i-j| in {1,2,3,5}.
# We want a clique of size 12.
# Let's try to build one.
# 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13
# Check 0 and 5: diff 5 (ok)
# 0 and 6: diff 6 (no)
# So 0 and 6 can't both be in the set.
# Let's try: 0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13, 14?
# 0-5 (5), 0-7 (7, no).
# This is getting complex. Let's just do a search.

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Initial solution: random
    best_arr = [random.randint(1, 10000) for _ in range(12)]
    # Ensure distinct
    while len(set(best_arr)) < 12:
        best_arr[random.randint(0, 11)] = random.randint(1, 10000)
    best_cost = count_distinct_primes_fast(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.9995
    
    # Precompute a list of "good" values? 
    # Maybe values that are powers of 2, or 2*3^k, etc.
    # Let's just use random for now, but bias towards small numbers.
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Pick 2 indices to swap or modify
        # Move 1: Change one element
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        
        # Generate new value
        # Try small values first, then random
        if random.random() < 0.5:
            new_val = random.randint(1, 1000)
        else:
            new_val = random.randint(1, 10000)
            
        if new_val in current_arr and new_val != old_val:
            continue
            
        current_arr[idx] = new_val
        new_cost = count_distinct_primes_fast(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
                # If we found a very good solution (e.g. < 5), we can try to refine it
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        
        # Move 2: Swap two elements with random values
        if random.random() < 0.2:
            i1, i2 = random.sample(range(12), 2)
            v1, v2 = current_arr[i1], current_arr[i2]
            
            new_v1 = random.randint(1, 10000)
            new_v2 = random.randint(1, 10000)
            
            # Check distinctness
            if (new_v1 in current_arr and new_v1 != v1) or (new_v2 in current_arr and new_v2 != v2) or new_v1 == new_v2:
                continue
                
            current_arr[i1] = new_v1
            current_arr[i2] = new_v2
            
            new_cost = count_distinct_primes_fast(current_arr)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_arr = current_arr[:]
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    current_arr[i1] = v1
                    current_arr[i2] = v2
        
        T *= cooling_rate
        
    return best_arr
