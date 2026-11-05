import time
import heapq

_START = time.time()
_DEADLINE = 80.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # best_size[r] = minimum subset size that sums to r mod MOD
    # best_mask[r] = the mask achieving that minimum size
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD
    
    # Dijkstra's algorithm on the state space of (sum, mask)
    # State: (cost, sum_mod, mask)
    # Cost is the number of elements (popcount of mask)
    # We start from single-element masks to ensure non-empty sets
    pq = []
    for x in range(1, N + 1):
        bit = 1 << (x - 1)
        heapq.heappush(pq, (1, inv[x] % MOD, bit))
        
    visited = set()
    
    while pq:
        size, s, m = heapq.heappop(pq)
        
        if m in visited:
            continue
        visited.add(m)
        
        # Update best for this sum
        if size < best_size[s]:
            best_size[s] = size
            best_mask[s] = m
            
        # Expand: add each unused denominator
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if not (m & bit):
                new_m = m | bit
                new_s = (s + inv[x]) % MOD
                new_size = size + 1
                heapq.heappush(pq, (new_size, new_s, new_m))
                
    # Construct the result
    # row r contains the denominators from best_mask[r]
    result = []
    for r in range(MOD):
        m = best_mask[r]
        terms = []
        for x in range(1, N + 1):
            if m & (1 << (x - 1)):
                terms.append(x)
        result.append(terms)
        
    return result
