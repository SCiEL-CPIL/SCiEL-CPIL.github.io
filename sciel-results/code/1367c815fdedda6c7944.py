"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 5000
DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.where(sieve)[0].tolist()


def construct():
    start_time = time.time()
    
    # All integers in [0, N]
    all_nums = np.arange(N + 1, dtype=np.int32)
    
    # Primes up to N+1 (since |S| can be up to N+1)
    primes = _primes(N + 1)
    
    # Track which numbers are still alive
    alive = np.ones(N + 1, dtype=bool)
    alive_count = N + 1
    
    # For each prime, we'll choose a residue class to ban
    # Greedy: for each prime p, ban the residue that removes the fewest alive elements
    # But we need p <= |S|, so we process primes in order and stop when p > alive_count
    
    # Sort primes - they already are
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
        if p > alive_count:
            break
        
        # For each residue class mod p, count how many alive elements fall in it
        # residues of alive elements mod p
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        residues = alive_indices % p
        # Count occurrences of each residue
        # Use bincount for speed
        counts = np.bincount(residues, minlength=p)
        
        # Find the residue with minimum count (ban it to remove fewest)
        min_residue = int(np.argmin(counts))
        
        # Remove all alive elements in this residue class
        to_remove = (residues == min_residue)
        alive[alive_indices[to_remove]] = False
        alive_count -= int(np.sum(to_remove))
    
    # Extract the surviving set
    result = np.where(alive)[0].tolist()
    
    # Verify admissibility and fix if needed
    # Check: for every prime p <= len(result), some residue class mod p is missed
    result_set = set(result)
    n = len(result)
    primes_check = _primes(n)
    
    for p in primes_check:
        if time.time() - start_time > DEADLINE:
            break
        # Check if all p residues are covered
        residues_covered = set(x % p for x in result_set)
        if len(residues_covered) == p:
            # Need to remove elements from one residue class
            # Find the residue class with fewest elements
            from collections import defaultdict
            class_members = defaultdict(list)
            for x in result_set:
                class_members[x % p].append(x)
            # Remove the smallest class
            min_class = min(class_members.values(), key=len)
            for x in min_class:
                result_set.discard(x)
            n = len(result_set)
            # Re-check primes up to new n
            # Restart check from beginning for correctness
            result_set = set(result_set)
            for p2 in _primes(n):
                residues_covered2 = set(x % p2 for x in result_set)
                if len(residues_covered2) == p2:
                    from collections import defaultdict
                    class_members2 = defaultdict(list)
                    for x in result_set:
                        class_members2[x % p2].append(x)
                    min_class2 = min(class_members2.values(), key=len)
                    for x in min_class2:
                        result_set.discard(x)
                    n = len(result_set)
    
    return sorted(result_set)
# EVOLVE-BLOCK-END
