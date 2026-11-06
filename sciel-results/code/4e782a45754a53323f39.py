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
    
    # Precompute residue coverage counts for each prime
    # coverage[p][r] = number of active points with residue r mod p
    prime_idx = {p: i for i, p in enumerate(primes)}
    coverage = []
    for p in primes:
        counts = [0] * p
        for x in current_set:
            counts[x % p] += 1
        coverage.append(counts)
    
    # Number of missed classes per prime
    missed = [sum(1 for c in cov if c == 0) for cov in coverage]
    
    # Local improvement: try to add back removed points one at a time
    removed = [x for x in range(N + 1) if not active[x]]
    
    # Sort removed by a heuristic: prefer points that are "less constraining"
    # A point x is less constraining if it has fewer "critical" residues
    # (residues that are currently the only one missed for some prime)
    # For each x, count how many primes p have exactly one missed class,
    # and x's residue mod p is NOT that missed class (so adding x won't help)
    # vs x's residue IS the missed class (adding x would fill it, potentially making it inadmissible)
    
    # Simpler heuristic: try points in order, but use incremental update
    # for the admissibility check
    
    # Precompute for each removed point x, which primes would become "all covered" if x is added
    # A prime p becomes problematic if missed[p] == 1 and x % p is the missed residue
    
    # Build a mapping: for each prime, which residue is the unique missed one (if any)
    unique_missed_res = [None] * len(primes)
    for i, cov in enumerate(coverage):
        zeros = [r for r, c in enumerate(cov) if c == 0]
        if len(zeros) == 1:
            unique_missed_res[i] = zeros[0]
    
    # For each removed point, count how many primes would become fully covered
    point_cost = {}
    for x in removed:
        cost = 0
        for i, p in enumerate(primes):
            if unique_missed_res[i] is not None and x % p == unique_missed_res[i]:
                cost += 1
        point_cost[x] = cost
    
    # Sort by cost (lower cost = safer to add)
    removed_sorted = sorted(removed, key=lambda x: point_cost[x])
    
    # Try adding points back in sorted order
    for x in removed_sorted:
        if time.time() > limit_time:
            break
        
        new_size = active_count + 1
        
        # Check if adding x back keeps the set admissible
        # For each prime p <= new_size, check if some residue class mod p is still missed
        valid = True
        for i, p in enumerate(primes):
            if p > new_size:
                break
            
            r = x % p
            # After adding x, coverage[i][r] increases by 1
            # If coverage[i][r] was 0, it becomes 1, so missed count decreases by 1
            # If coverage[i][r] was > 0, missed count stays the same
            
            old_missed = missed[i]
            if coverage[i][r] == 0:
                new_missed = old_missed - 1
            else:
                new_missed = old_missed
            
            if new_missed == 0:
                # All classes would be covered -> not admissible
                valid = False
                break
        
        if valid:
            active[x] = True
            active_count += 1
            current_set.append(x)
            
            # Update coverage and missed counts
            for i, p in enumerate(primes):
                r = x % p
                coverage[i][r] += 1
                if coverage[i][r] == 1:  # was 0, now 1
                    missed[i] -= 1
                # Update unique_missed_res if needed
                if missed[i] <= 1:
                    zeros = [r2 for r2, c in enumerate(coverage[i]) if c == 0]
                    if len(zeros) == 1:
                        unique_missed_res[i] = zeros[0]
                    else:
                        unique_missed_res[i] = None
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
