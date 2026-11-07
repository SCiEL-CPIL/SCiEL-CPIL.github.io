import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
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
    
    # For each prime, the standalone value is the largest power p^k <= n
    standalone_val = {}
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        standalone_val[p] = val
    
    # Calculate merged values for pairs of primes
    # For primes p, q, the best merged value is max p^a * q^b <= n
    # The gain is merged_val - standalone_val[p] - standalone_val[q]
    pairs = {}
    for i in range(len(primes)):
        p = primes[i]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p * q > n:
                break
            
            # Find max product p^a q^b <= n
            max_prod = 0
            max_a = int(math.log(n, p)) + 1
            for a in range(1, max_a + 1):
                pa = p ** a
                if pa > n:
                    break
                max_b = int(math.log(n / pa, q)) + 1 if n / pa >= 1 else 0
                for b in range(1, max_b + 1):
                    prod = pa * (q ** b)
                    if prod <= n and prod > max_prod:
                        max_prod = prod
            
            if max_prod > 0:
                gain = max_prod - standalone_val[p] - standalone_val[q]
                if gain > 0:
                    pairs[(p, q)] = (max_prod, gain)
    
    # Build adjacency list for active primes (those with positive gain pairs)
    active_primes = set()
    adj = {p: [] for p in primes}
    for (p, q), (val, gain) in pairs.items():
        active_primes.add(p)
        active_primes.add(q)
        adj[p].append((q, gain, val))
        adj[q].append((p, gain, val))
    
    # Primes not in active_primes have no beneficial merges, so they contribute standalone_val[p]
    base_sum = sum(standalone_val[p] for p in primes if p not in active_primes)
    # Also add 1 to the base sum
    base_sum += 1
    
    # Now we need to find the maximum weight matching on the graph of active_primes
    # where edge weights are the gains
    # This is a maximum weight matching problem on a general graph
    
    active_list = sorted(list(active_primes))
    n_active = len(active_list)
    
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(active_list)}
    
    # Build adjacency in terms of indices
    adj_idx = [[] for _ in range(n_active)]
    edge_weight = {}  # (i, j) -> (gain, merged_val)
    for (p, q), (val, gain) in pairs.items():
        i = prime_to_idx[p]
        j = prime_to_idx[q]
        adj_idx[i].append(j)
        adj_idx[j].append(i)
        edge_weight[(i, j)] = (gain, val)
        edge_weight[(j, i)] = (gain, val)
    
    # Use branch and bound for maximum weight matching
    # State: set of available vertices (as a frozenset of indices)
    # We use memoization
    
    import sys
    sys.setrecursionlimit(10000)
    
    cache = {}
    
    def solve(available):
        if time.time() > deadline:
            return 0, []
        if not available:
            return 0, []
        key = tuple(sorted(available))
        if key in cache:
            return cache[key]
        
        # Pick the vertex with smallest index
        v = min(available)
        remaining = list(available)
        remaining.remove(v)
        
        # Option 1: don't match v
        best_gain, best_edges = solve(frozenset(remaining))
        
        # Option 2: match v with a neighbor
        for u in adj_idx[v]:
            if u in available:
                new_remaining = list(remaining)
                new_remaining.remove(u)
                g, edges = solve(frozenset(new_remaining))
                gain_here = edge_weight[(v, u)][0] + g
                if gain_here > best_gain:
                    best_gain = gain_here
                    best_edges = edges + [(v, u)]
        
        result = (best_gain, best_edges)
        cache[key] = result
        return result
    
    max_gain, chosen_edges = solve(frozenset(range(n_active)))
    
    # Reconstruct the result
    result = [1]
    used_primes = set()
    
    for v, u in chosen_edges:
        p = active_list[v]
        q = active_list[u]
        val = edge_weight[(v, u)][1]
        result.append(val)
        used_primes.add(p)
        used_primes.add(q)
    
    for p in primes:
        if p not in used_primes:
            result.append(standalone_val[p])
    
    # Ensure all values are distinct and in range
    # Check for duplicates
    seen = set()
    final_result = []
    for x in result:
        if x not in seen and 1 <= x <= 256:
            final_result.append(x)
            seen.add(x)
    
    # Verify pairwise coprime
    def is_valid(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    if not is_valid(final_result):
        # Fallback: just use standalone primes + 1
        final_result = [1] + [standalone_val[p] for p in primes]
    
    final_result = [int(x) for x in final_result]
    
    return final_result
