import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    # Precompute divisors and prime powers
    is_prime_power = [False] * (n + 1)
    prime_power_base = [0] * (n + 1)
    for x in range(2, n + 1):
        # Check if x is a prime power
        found = False
        for p in range(2, int(math.isqrt(x)) + 1):
            if x % p == 0:
                y = x
                while y % p == 0:
                    y //= p
                if y == 1:
                    is_prime_power[x] = True
                    prime_power_base[x] = p
                    found = True
                    break
        if not found:
            if x > 1 and all(x % p != 0 for p in range(2, int(math.isqrt(x)) + 1)):
                is_prime_power[x] = True
                prime_power_base[x] = x

    def has_conflict(u, v):
        if u > v:
            u, v = v, u
        
        # Case 1: Neither divides the other
        if u % v != 0 and v % u != 0:
            l = u * v // math.gcd(u, v)
            if l <= n:
                return True
            else:
                return False
        
        # Case 2: u divides v (u < v)
        # v is multiple of u
        ratio = v // u
        # Check if ratio is prime and u is prime power of that prime
        if is_prime_power[ratio]:
            p = prime_power_base[ratio]
            if is_prime_power[u] and prime_power_base[u] == p:
                # Only solutions are u and v
                return False
        return True

    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if has_conflict(i, j):
                adj[i].append(j)
                adj[j].append(i)
                
    # Find Max Independent Set using Bron-Kerbosch on complement graph?
    # Or just use a simple heuristic/backtracking since N=128 is small.
    # Actually, MIS is NP-hard, but for N=128 and sparse graph, we can use a greedy + local search.
    
    # Greedy initialization: sort by degree ascending
    degrees = [len(adj[i]) for i in range(1, n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: degrees[x])
    
    best_set = set()
    current_set = set()
    
    for v in order:
        # Check if v can be added
        conflict = False
        for u in current_set:
            if u in adj[v]:
                conflict = True
                break
        if not conflict:
            current_set.add(v)
            
    best_set = set(current_set)
    
    # Local search: try to add nodes, then swap
    # Since we want to maximize size, we can try to add any node that fits.
    # The greedy above already did this in a specific order.
    # Let's try a few random restarts or different orderings if time permits.
    
    import random
    random.seed(42)
    
    for _ in range(50): # Fixed number of restarts
        if time.time() > deadline:
            break
        random.shuffle(order)
        current_set = set()
        for v in order:
            conflict = False
            for u in current_set:
                if u in adj[v]:
                    conflict = True
                    break
            if not conflict:
                current_set.add(v)
        
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
        # Local improvement: try to add nodes that were skipped
        # This is tricky to implement efficiently without re-checking all pairs.
        # Given the small size, the greedy with random ordering is often sufficient.
        
    return sorted(best_set)
