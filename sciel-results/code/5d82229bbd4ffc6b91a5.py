"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    import time
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Start with all integers in [0, N]
    points = set(range(N + 1))
    
    # Generate primes up to N
    primes = _primes(N)
    
    # We need to ensure that for every prime p <= len(points), not all residues are covered.
    # As we remove points, len(points) decreases.
    # If p > len(points), it is impossible to cover all p residues, so the condition is automatically satisfied.
    # Thus, we only need to process primes p where p <= len(points).
    
    for p in primes:
        if time.time() > deadline:
            break
            
        current_len = len(points)
        if p > current_len:
            break
            
        # Check if all residues mod p are present
        # We can do this by counting residues.
        # Since we need to find the least populated class to remove, we must count anyway.
        residue_counts = [0] * p
        residue_members = [set() for _ in range(p)]
        
        for v in points:
            r = v % p
            residue_counts[r] += 1
            residue_members[r].add(v)
            
        # Check if any residue is missing
        if 0 in residue_counts:
            # Admissibility condition satisfied for this p
            continue
            
        # Find the residue class with the minimum count
        min_count = min(residue_counts)
        # There might be multiple classes with the same min count.
        # Removing any of them will satisfy the condition for this p.
        # However, removing a class might affect future primes.
        # The "greedy" strategy is to remove the smallest class.
        # If there's a tie, does it matter?
        # Usually, removing the smallest is optimal for minimizing loss.
        
        # Remove all points in the smallest residue class(es)?
        # The condition is "some residue class is missed".
        # So we only need to empty ONE class.
        # We should pick the one with the fewest elements to minimize the reduction in set size.
        
        min_residue = -1
        min_size = float('inf')
        for r in range(p):
            if residue_counts[r] < min_size:
                min_size = residue_counts[r]
                min_residue = r
                # If we found a class with size 0, we would have continued.
                # So min_size >= 1.
                
        # Remove points in min_residue
        points.difference_update(residue_members[min_residue])
        
    return sorted(points)
# EVOLVE-BLOCK-END
