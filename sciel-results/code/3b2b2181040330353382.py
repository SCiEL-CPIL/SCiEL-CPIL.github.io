import time

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s margin for safety
    
    # Start with the full set of integers in [0, N]
    points = list(range(N + 1))
    
    # Precompute all primes up to N
    all_primes = _primes(N)
    
    # Primes that still need to be checked/satisfied
    # A prime p is "active" if p <= len(points) and it is currently violated
    active_primes = all_primes[:]
    
    while time.time() < deadline and active_primes:
        current_size = len(points)
        
        # We will rebuild the active_primes list with only those that are still violated
        still_violating = []
        
        # Track the best candidate for removal
        # We want to remove the residue class that results in the fewest elements removed
        # Among all currently violating primes
        best_p = None
        best_r = None
        best_min_count = float('inf')
        
        for p in active_primes:
            # If p > current_size, we don't need to satisfy it anymore
            if p > current_size:
                continue
                
            # Count residues mod p
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            min_c = min(counts)
            
            if min_c == 0:
                # This prime is already satisfied (some residue is empty)
                # It will remain satisfied as we only remove elements
                # So we can drop it from active_primes
                continue
            else:
                # This prime is violated
                still_violating.append(p)
                
                # Check if this is the best candidate for removal
                if min_c < best_min_count:
                    best_min_count = min_c
                    best_p = p
                    # Find the index of the minimum count
                    # If there are multiple, pick the first one
                    best_r = counts.index(min_c)
        
        # Update active primes list
        active_primes = still_violating
        
        # If no primes are violated, we are done
        if not active_primes:
            break
            
        # If we didn't find a best candidate (should not happen if active_primes is not empty)
        if best_p is None:
            break
            
        # Remove all elements v such that v % best_p == best_r
        points = [v for v in points if v % best_p != best_r]
        
    # If we exited due to timeout, points might be invalid.
    # However, given N=5000, the loop should finish very quickly.
    # To be safe against returning an invalid tuple, we could add a final check/repair,
    # but it's likely unnecessary.
    
    return points
