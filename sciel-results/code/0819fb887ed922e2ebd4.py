import time

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def construct():
    # Combine the "remove 0 mod p" structural strategy from the current program
    # with the efficient repair machinery from the inspirations.
    # 
    # The key insight is that the "remove 0 mod p" strategy yields a set S_K
    # where K is the threshold prime. This set is admissible if K >= |S_K|.
    # The size |S_K| is invariant to the choice of residues removed (for a fixed
    # set of primes), so "remove 0" is optimal in terms of final size for a given K.
    # 
    # However, the inspirations show that a greedy repair approach (starting from
    # the full set and removing the least-populated residue class for each prime
    # that is fully covered) can achieve a slightly better score (1.0176 vs 0.9042).
    # 
    # To combine these: we use the efficient boolean-array based counting from the
    # inspirations to speed up the greedy repair, but we initialize the search with
    # the "remove 0 mod p" structure to get a good starting point, then refine.
    # 
    # Actually, the inspirations' greedy repair starts from the full set [0, N] and
    # repairs. The current program's "remove 0 mod p" is a specific instance of a
    # valid admissible set. The greedy repair from the full set is more general and
    # finds the largest admissible set by removing the minimum necessary elements.
    # 
    # The best approach is to use the greedy repair strategy from the inspirations,
    # but with the performance optimizations (boolean arrays, efficient counting)
    # that were measured to improve the score from 1.005 to 1.018.
    
    points = list(range(N + 1))
    primes = _primes(N)
    
    # Use a boolean array to track which points are still in the set
    # This allows O(1) removal and efficient iteration
    in_set = [True] * (N + 1)
    current_len = N + 1
    
    while time.time() < _DEADLINE:
        k = current_len
        if k < 2:
            break
        
        # Find the smallest prime p <= k such that points covers all residues mod p
        removed = False
        for p in primes:
            if p > k:
                break
            
            # Count residues using efficient iteration over active points
            counts = [0] * p
            for v in range(N + 1):
                if in_set[v]:
                    counts[v % p] += 1
            
            min_count = min(counts)
            if min_count > 0:
                # All residues are present. We must remove one residue class.
                # Remove the one with the fewest elements to maximize remaining size.
                target_residue = counts.index(min_count)
                
                # Remove all elements congruent to target_residue mod p
                for v in range(target_residue, N + 1, p):
                    if in_set[v]:
                        in_set[v] = False
                        current_len -= 1
                
                removed = True
                break
        
        if not removed:
            break
    
    # Build the final list
    result = [v for v in range(N + 1) if in_set[v]]
    return result
# EVOLVE-BLOCK-END
