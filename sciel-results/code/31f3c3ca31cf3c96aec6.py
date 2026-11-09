import time
import random
import math

MAX_SUM = 20000

# Precompute smallest prime factor for numbers up to MAX_SUM
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i * i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def _count_distinct_primes(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= factors_cache[s]
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr)
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Strategy 1: Bitset-based clique search for smooth sums
    # This is much faster than list-based intersection and can find large cliques
    prime_sets_to_try = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    for primes_set in prime_sets_to_try:
        if time.time() > deadline:
            break
        if len(primes_set) >= best_count:
            continue
        
        # Build bitsets for each candidate
        candidates = []
        for x in range(1, 10001):
            if all(p not in factors_cache[2 * x] or factors_cache[2 * x].issubset(set(primes_set)) for p in [2]):
                # Check if 2*x is smooth with respect to primes_set
                if factors_cache[2 * x].issubset(set(primes_set)):
                    candidates.append(x)
        
        if len(candidates) < 12:
            continue
        
        # Build adjacency bitsets
        n_cand = len(candidates)
        adj = [0] * n_cand
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                if factors_cache[candidates[i] + candidates[j]].issubset(set(primes_set)):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Greedy clique search with multiple starting points
        best_clique = []
        for start_idx in range(min(n_cand, 200)):
            if time.time() > deadline:
                break
            clique = [start_idx]
            available = adj[start_idx]
            while available:
                # Pick the vertex with the highest degree in the available set
                best_v = -1
                best_deg = -1
                temp = available
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = bin(adj[v] & available).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                clique.append(best_v)
                available &= adj[best_v]
                if len(clique) >= 12:
                    break
            
            if len(clique) > len(best_clique):
                best_clique = clique
                if len(best_clique) >= 12:
                    break
        
        if len(best_clique) >= 12:
            arr = [candidates[i] for i in best_clique[:12]]
            count = _count_distinct_primes(arr)
            if count < best_count:
                best_count = count
                best_arr = arr
                if best_count <= 3:
                    return best_arr
    
    # Strategy 2: Residue class search
    for modulus in range(2, 1000):
        if time.time() > deadline:
            break
        for residue in range(min(modulus, 10)):
            start = residue
            if start == 0:
                start = modulus
            if start > 10000:
                continue
            arr = []
            val = start
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += modulus
            if len(arr) < 12:
                continue
            count = _count_distinct_primes(arr)
            if count < best_count:
                best_count = count
                best_arr = arr
                if best_count <= 3:
                    return best_arr
    
    # Strategy 3: Random search with local improvement
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 20000:
        attempts += 1
        arr = random.sample(range(1, 10001), 12)
        count = _count_distinct_primes(arr)
        if count < best_count:
            best_count = count
            best_arr = arr
            if best_count <= 3:
                return best_arr
        else:
            improved = True
            local_count = count
            while improved and time.time() < deadline:
                improved = False
                for i in range(12):
                    for _ in range(3):
                        new_val = random.randint(1, 10000)
                        if new_val in arr and new_val != arr[i]:
                            continue
                        new_arr = arr.copy()
                        new_arr[i] = new_val
                        if len(set(new_arr)) != 12:
                            continue
                        new_count = _count_distinct_primes(new_arr)
                        if new_count < local_count:
                            local_count = new_count
                            arr = new_arr
                            improved = True
                            if local_count < best_count:
                                best_count = local_count
                                best_arr = arr
                                if best_count <= 3:
                                    return best_arr
                            break
    
    return best_arr
