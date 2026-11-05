"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import heapq
from collections import defaultdict

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
    
    # For each prime, we track which residue classes are still "present" 
    # (i.e., at least one number in the current set has that residue).
    # We want to remove numbers such that for every prime p <= len(tuple),
    # at least one residue class mod p is completely missing.
    
    # Greedy approach: iteratively remove the number that "covers" the most 
    # uncovered constraints. But actually, we want to minimize removals while 
    # satisfying all constraints.
    
    # Alternative greedy: For each prime p, we need to ensure at least one 
    # residue class is empty. We can choose which residue class to empty.
    # The cost of emptying residue r mod p is the number of active elements 
    # with x % p == r.
    
    # We use a priority queue to always pick the cheapest residue class to 
    # empty among the most critical primes.
    
    # Precompute for each prime, the list of elements in each residue class
    residue_elements = []
    for p in primes:
        classes = [[] for _ in range(p)]
        for x in range(N + 1):
            classes[x % p].append(x)
        residue_elements.append(classes)
    
    # Initial counts: number of active elements in each residue class for each prime
    counts = []
    for i, p in enumerate(primes):
        c = [len(residue_elements[i][r]) for r in range(p)]
        counts.append(c)
    
    active = [True] * (N + 1)
    current_size = N + 1
    satisfied = [False] * num_primes
    
    # Heap of (cost, prime_idx, residue_idx)
    heap = []
    for i in range(num_primes):
        min_cost = min(counts[i])
        min_r = counts[i].index(min_cost)
        heapq.heappush(heap, (min_cost, i, min_r))
    
    while heap:
        cost, p_idx, r_idx = heapq.heappop(heap)
        p = primes[p_idx]
        
        # Check if this prime is still relevant (p <= current_size)
        if p > current_size:
            continue
        
        # Check if satisfied
        if satisfied[p_idx]:
            continue
        
        # Get current min count for this prime
        cur_min = min(counts[p_idx])
        
        if cur_min == 0:
            satisfied[p_idx] = True
            continue
        
        # If the cost in heap is not the current min, it's stale
        if cur_min != cost:
            heapq.heappush(heap, (cur_min, p_idx, r_idx))
            continue
        
        # Verify r_idx is still a minimum
        if counts[p_idx][r_idx] != cost:
            heapq.heappush(heap, (cur_min, p_idx, r_idx))
            continue
        
        # Remove all active elements in this residue class
        to_remove = residue_elements[p_idx][r_idx]
        
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
