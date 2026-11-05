"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import heapq

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    primes = _primes(N)
    num_primes = len(primes)
    
    # Precompute elements for each residue class of each prime
    # residue_elements[i][r] = list of x in [0, N] such that x % primes[i] == r
    residue_elements = []
    for p in primes:
        classes = [[] for _ in range(p)]
        for x in range(N + 1):
            classes[x % p].append(x)
        residue_elements.append(classes)
        
    # Initialize counts
    # counts[i][r] = number of active x in residue_elements[i][r]
    counts = []
    for i, p in enumerate(primes):
        c = [0] * p
        for r in range(p):
            c[r] = len(residue_elements[i][r])
        counts.append(c)
        
    active = [True] * (N + 1)
    current_size = N + 1
    satisfied = [False] * num_primes
    
    # Heap of (cost, prime_idx)
    heap = []
    for i in range(num_primes):
        min_c = min(counts[i])
        heapq.heappush(heap, (min_c, i))
        
    while heap:
        cost, p_idx = heapq.heappop(heap)
        p = primes[p_idx]
        
        # Check if this prime is still relevant (p <= current_size)
        if p > current_size:
            continue
            
        # Check if satisfied
        if satisfied[p_idx]:
            continue
            
        # Get current min count
        cur_min = min(counts[p_idx])
        
        if cur_min == 0:
            satisfied[p_idx] = True
            continue
            
        # If the cost in heap is not the current min, it's stale
        if cur_min != cost:
            heapq.heappush(heap, (cur_min, p_idx))
            continue
            
        # This is the best prime to fix
        # Find the residue with min count
        r = counts[p_idx].index(cost)
        
        # Remove all elements in this residue class
        to_remove = residue_elements[p_idx][r]
        
        for x in to_remove:
            if active[x]:
                active[x] = False
                current_size -= 1
                
                # Update counts for all primes
                for j in range(num_primes):
                    pj = primes[j]
                    rj = x % pj
                    counts[j][rj] -= 1
        
        satisfied[p_idx] = True
        
    return [x for x in range(N + 1) if active[x]]
# EVOLVE-BLOCK-END
