import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s safety margin within 60s limit
    
    n = 256
    
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Calculate standalone value for each prime (highest power <= n)
    standalone_val = {}
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        standalone_val[p] = val
    
    # Precompute powers of each prime
    prime_powers = {}
    for p in primes:
        powers = []
        val = p
        while val <= n:
            powers.append(val)
            val *= p
        prime_powers[p] = powers
    
    # Check if any pair merge yields positive gain
    # For each pair (p, q), find max p^a * q^b <= n, compute gain
    has_positive_gain = False
    for i in range(len(primes)):
        p = primes[i]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p * q > n:
                break
            max_prod = 0
            for pa in prime_powers[p]:
                if pa * q > n:
                    break
                for qa in prime_powers[q]:
                    prod = pa * qa
                    if prod > n:
                        break
                    if prod > max_prod:
                        max_prod = prod
            if max_prod > 0:
                gain = max_prod - standalone_val[p] - standalone_val[q]
                if gain > 0:
                    has_positive_gain = True
                    break
        if has_positive_gain:
            break
    
    if not has_positive_gain:
        # Optimal: use 1 and the highest power of each prime
        result = [1] + [standalone_val[p] for p in primes]
        return [int(x) for x in result]
    
    # If positive gains exist, use matching
    pairs = {}
    for i in range(len(primes)):
        p = primes[i]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p * q > n:
                break
            max_prod = 0
            for pa in prime_powers[p]:
                if pa * q > n:
                    break
                for qa in prime_powers[q]:
                    prod = pa * qa
                    if prod > n:
                        break
                    if prod > max_prod:
                        max_prod = prod
            if max_prod > 0:
                gain = max_prod - standalone_val[p] - standalone_val[q]
                if gain > 0:
                    pairs[(p, q)] = (max_prod, gain)
    
    active_primes = set()
    adj = {p: [] for p in primes}
    for (p, q), (val, gain) in pairs.items():
        active_primes.add(p)
        active_primes.add(q)
        adj[p].append((q, gain, val))
        adj[q].append((p, gain, val))
    
    active_list = sorted(list(active_primes))
    prime_to_idx = {p: i for i, p in enumerate(active_list)}
    num_active = len(active_list)
    
    adj_idx = [[] for _ in range(num_active)]
    for p in active_list:
        idx = prime_to_idx[p]
        for q, gain, val in adj[p]:
            if q in prime_to_idx:
                adj_idx[idx].append((prime_to_idx[q], gain, val))
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def solve_with_pairs(mask):
        if mask == 0:
            return 0, []
        p_idx = (mask & -mask).bit_length() - 1
        best_gain = -1
        best_pairs = []
        g1, pairs1 = solve_with_pairs(mask ^ (1 << p_idx))
        if g1 > best_gain:
            best_gain = g1
            best_pairs = pairs1
        for q_idx, gain, val in adj_idx[p_idx]:
            if mask & (1 << q_idx):
                new_mask = mask ^ (1 << p_idx) ^ (1 << q_idx)
                g2, pairs2 = solve_with_pairs(new_mask)
                total_gain = gain + g2
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_pairs = pairs2 + [(active_list[p_idx], active_list[q_idx], val)]
        return best_gain, best_pairs
    
    full_mask = (1 << num_active) - 1
    
    # Check time before expensive computation
    if time.time() > deadline:
        # Fallback: just use standalone values
        result = [1] + [standalone_val[p] for p in primes]
        return [int(x) for x in result]
    
    _, chosen_pairs = solve_with_pairs(full_mask)
    
    result = [1]
    used_primes = set()
    for p, q, val in chosen_pairs:
        result.append(val)
        used_primes.add(p)
        used_primes.add(q)
    for p in primes:
        if p not in used_primes:
            result.append(standalone_val[p])
    
    return [int(x) for x in result]
