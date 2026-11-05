import time
import random
import math
import heapq
from collections import defaultdict

MAX_VAL = 20000
MAX_NODE = 10000

# Precompute smallest prime factor for fast factorization
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

def generate_smooth(primes, limit):
    """Generate all primes-smooth numbers up to limit using a heap."""
    if not primes:
        return set()
    
    smooth = set([1])
    heap = [1]
    heapq.heapify(heap)
    
    while heap:
        val = heapq.heappop(heap)
        if val > limit:
            break
        smooth.add(val)
        for p in primes:
            new_val = val * p
            if new_val <= limit:
                heapq.heappush(heap, new_val)
                
    return smooth

def find_clique_bitset(adj_bits, target_size, deadline):
    """
    Find a clique of size target_size using bitset operations.
    adj_bits[i] is an integer bitmask where bit j is 1 if (i,j) is an edge.
    """
    best_clique = []
    
    def dfs(candidates_mask, current_clique):
        nonlocal best_clique
        if time.time() > deadline:
            return False
        
        if len(current_clique) == target_size:
            best_clique = current_clique[:]
            return True
        
        if candidates_mask == 0:
            return False
            
        # Count bits in candidates_mask
        remaining = bin(candidates_mask).count('1')
        if remaining < target_size - len(current_clique):
            return False
            
        # Try nodes in order of increasing index
        mask = candidates_mask
        while mask:
            # Get lowest set bit
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            
            if v < 1 or v > MAX_NODE:
                mask ^= lsb
                continue
                
            # Check if v is connected to all in current_clique
            connected = True
            for u in current_clique:
                if not (adj_bits[u] >> v) & 1:
                    connected = False
                    break
            
            if not connected:
                mask ^= lsb
                continue
            
            new_clique = current_clique + [v]
            
            # New candidates: must be in candidates_mask without v, and connected to v
            new_candidates = (candidates_mask & ~lsb) & adj_bits[v]
            
            if dfs(new_candidates, new_clique):
                return True
                
            mask ^= lsb
            
        return False
    
    # Initial candidates: all nodes 1..MAX_NODE
    initial_mask = 0
    for i in range(1, MAX_NODE + 1):
        initial_mask |= (1 << i)
    
    dfs(initial_mask, [])
    
    return best_clique if best_clique else None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try prime sets in order of size
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    for P in prime_sets:
        if time.time() > deadline - 5:
            break
            
        smooth_sums = generate_smooth(P, 20000)
        
        # Build adjacency bitmasks
        adj_bits = [0] * (MAX_NODE + 1)
        
        for x in range(1, MAX_NODE + 1):
            mask = 0
            for s in smooth_sums:
                y = s - x
                if 1 <= y <= MAX_NODE and y != x:
                    mask |= (1 << y)
            adj_bits[x] = mask
        
        # Find clique
        clique = find_clique_bitset(adj_bits, 12, deadline - 2)
        
        if clique:
            cost = count_distinct_primes(clique)
            if cost <= len(P):
                return clique
    
    # Fallback: Simulated Annealing
    best_arr = [3 + 6 * k for k in range(12)]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < 0.01:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
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
                current_arr[idx] = old_val
        
        T *= cooling_rate
    
    return best_arr
