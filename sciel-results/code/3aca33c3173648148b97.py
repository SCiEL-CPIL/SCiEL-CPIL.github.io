import math
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 290.0

    primes = _primes(N)
    
    # Start with all points
    active = [True] * (N + 1)
    active_count = N + 1
    
    # Greedy: for each prime p, if all residue classes are present, remove the smallest class
    for p in primes:
        if time.time() > limit_time:
            break
        if p > active_count:
            break
        
        # Count active points in each residue class mod p
        residue_counts = [0] * p
        for x in range(N + 1):
            if active[x]:
                residue_counts[x % p] += 1
        
        if min(residue_counts) > 0:
            # Find class with fewest active points
            min_idx = min(range(p), key=lambda i: residue_counts[i])
            # Remove all active points in this class
            for x in range(min_idx, N + 1, p):
                if active[x]:
                    active[x] = False
                    active_count -= 1
    
    # Build current set
    current_set = [x for x in range(N + 1) if active[x]]
    
    # Local improvement: try to add back removed points one at a time
    # A point x can be added back if for every prime p <= len(new_set), 
    # there still exists a missed residue class mod p
    removed = [x for x in range(N + 1) if not active[x]]
    
    # Sort removed by a heuristic: prefer points that are "less constraining"
    # i.e., points whose addition would violate fewer prime constraints
    # Simple heuristic: try in order of x
    for x in removed:
        if time.time() > limit_time:
            break
        
        new_size = active_count + 1
        if new_size > N + 1:
            break
        
        # Check if adding x back keeps the set admissible
        # For each prime p <= new_size, check if some residue class mod p is still missed
        valid = True
        for p in primes:
            if p > new_size:
                break
            # Check if all residue classes mod p are covered in current_set + {x}
            # If all are covered, adding x makes it inadmissible (unless x fills a gap)
            # Actually: we need that SOME class is missed. Currently some class is missed.
            # Adding x can only fill a missed class, making it potentially all-covered.
            # So check: after adding x, is there still a missed class mod p?
            
            # Count residues in current_set + {x}
            covered = [0] * p
            for y in current_set:
                covered[y % p] += 1
            covered[x % p] += 1
            
            if min(covered) > 0:
                # All classes covered -> not admissible
                valid = False
                break
        
        if valid:
            active[x] = True
            active_count += 1
            current_set.append(x)
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
