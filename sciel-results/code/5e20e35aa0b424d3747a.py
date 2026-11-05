import time
import random
import math
import heapq

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

def find_clique(adj_bits, target_size):
    """
    Find a clique of size target_size in the graph defined by adj_bits.
    adj_bits[i] is an integer bitmask where bit j is 1 if (i,j) is an edge.
    Nodes are 0..MAX_NODE. We use 1..MAX_NODE.
    """
    # Initial candidates: all nodes 1..MAX_NODE
    # We sort them by index to enforce ordering and avoid permutations
    candidates = list(range(1, MAX_NODE + 1))
    
    def dfs(candidates, current_clique):
        if len(current_clique) == target_size:
            return current_clique[:]
        
        if not candidates:
            return None
            
        # Pruning: not enough candidates
        if len(candidates) < target_size - len(current_clique):
            return None
            
        # Filter candidates that are connected to all nodes in current_clique
        if current_clique:
            mask = adj_bits[current_clique[0]]
            for u in current_clique[1:]:
                mask &= adj_bits[u]
        else:
            mask = (1 << (MAX_NODE + 1)) - 1 # All bits set
            
        valid_candidates = []
        for v in candidates:
            if (mask >> v) & 1:
                valid_candidates.append(v)
                
        if len(valid_candidates) < target_size - len(current_clique):
            return None
            
        # Heuristic: Try nodes with higher degree in the valid set first?
        # Or just iterate in increasing order (already sorted).
        # To speed up, we can pick the one with highest degree in valid_candidates.
        # But calculating degree for each step is costly.
        # Let's just iterate. The ordering by index is crucial for correctness (avoiding perms).
        
        for i, v in enumerate(valid_candidates):
            # New clique
            new_clique = current_clique + [v]
            
            # New candidates: must be in valid_candidates after i, and connected to v
            # Since valid_candidates is sorted, valid_candidates[i+1:] are all > v.
            # We need to intersect with adj_bits[v].
            new_candidates = []
            adj_v = adj_bits[v]
            for u in valid_candidates[i+1:]:
                if (adj_v >> u) & 1:
                    new_candidates.append(u)
                    
            res = dfs(new_candidates, new_clique)
            if res:
                return res
                
        return None
        
    return dfs(candidates, [])

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
        # adj_bits[x] has bit y set if x+y is in smooth_sums and x!=y
        adj_bits = [0] * (MAX_NODE + 1)
        
        for x in range(1, MAX_NODE + 1):
            mask = 0
            # y must be such that 1 <= y <= 10000, y != x
            # x + y in smooth_sums => y in smooth_sums - x
            for s in smooth_sums:
                y = s - x
                if 1 <= y <= MAX_NODE and y != x:
                    mask |= (1 << y)
            adj_bits[x] = mask
            
        # Find clique
        clique = find_clique(adj_bits, 12)
        
        if clique:
            # Verify
            cost = count_distinct_primes(clique)
            if cost <= len(P):
                return clique
    
    # Fallback: Simulated Annealing
    # Start with a decent initial guess
    # Multiples of 6: 6, 12, 18... Sums are 12, 18, 24... = 6*(2,3,4...).
    # Primes of sums: 2, 3. Plus primes of (k1+k2).
    # If k1+k2 is 2,3-smooth, total primes are {2,3}.
    # 1..12 sums: 3..23.
    # 1+2=3, 1+3=4, 1+4=5 (5 is prime).
    # So simple multiples of 6 don't guarantee {2,3} only.
    
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
