"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return all primes <= limit in increasing order."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_score(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has at least 2 hits."""
    # For each integer x in 1..cap, count how many residues hit it
    # x is hit by prime i if x % primes[i] == residues[i]
    # We need the first x where count < 2
    n = len(primes)
    # Use an array to accumulate counts
    # For efficiency, we process in chunks
    max_check = min(cap, max(primes) * 2 + 100)  # reasonable upper bound
    counts = [0] * (max_check + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        # x ≡ r (mod p), so x = r, r+p, r+2p, ...
        # x must be >= 1
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, max_check + 1, p):
            counts[x] += 1
    # Find the first gap
    for x in range(1, max_check + 1):
        if counts[x] < 2:
            return x - 1
    return max_check

def _greedy_construct(primes, deadline=None):
    """
    Forward greedy: assign residues one prime at a time.
    At step i (0-indexed), we have assigned residues for primes[0..i-1].
    We want to maximize the doubly-covered prefix length.
    
    Strategy: For each new prime, try all residues and pick the one that
    maximizes the resulting score, with tie-breaking favoring residues that
    cover the current frontier.
    
    To make this tractable, we use a heuristic: maintain a coverage array
    for a window around the current frontier, and for each candidate residue,
    compute how it affects coverage in that window.
    """
    n = len(primes)
    residues = [0] * n
    
    # We'll track coverage counts for integers in a growing window
    # Start with a small window and grow it
    # Key insight: with i primes assigned, the max doubly-covered prefix is at most i
    # (each prime can hit at most 1 integer in any block of size p, but we need 2 hits)
    # Actually, with i primes, we can cover at most i integers twice (each prime contributes 1 hit)
    # So the frontier grows by at most 1 per prime on average
    
    # Let's use a sliding window approach
    # Window: [1, W] where W grows as we add primes
    # For each candidate residue of prime i, compute the delta in coverage
    
    # Precompute: for each prime p, the set of integers in [1, W] that are ≡ r (mod p)
    # We'll use a coverage array and update it incrementally
    
    # Start: no primes assigned, coverage all 0
    # We'll maintain:
    #   cov[x] = number of assigned primes that hit x
    #   frontier = largest L such that cov[x] >= 2 for all x in 1..L
    
    # For efficiency, we only need to track coverage up to some bound
    # With 168 primes, the max score is at most 168 (if each prime adds 1 to the frontier)
    # But actually it could be more if some primes hit multiple frontier integers
    # Let's use a bound of 500 to be safe
    
    BOUND = 500
    cov = [0] * (BOUND + 1)
    frontier = 0  # largest L such that cov[x] >= 2 for all x in 1..L
    
    for i, p in enumerate(primes):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to find the best residue for prime i
        # Heuristic: we want to extend the frontier if possible
        # The frontier is at 'frontier', so we want cov[frontier+1] to become >= 2
        # Currently cov[frontier+1] = c. We need c + (1 if new residue hits frontier+1 else 0) >= 2
        # If c >= 1, any residue that hits frontier+1 works (and we can also try to extend further)
        # If c == 0, we need the new residue to hit frontier+1 AND we still need another hit
        
        # Better heuristic: for each candidate residue r in 0..p-1,
        #   compute the new frontier if we use r
        #   pick the r that gives the largest frontier
        #   tie-break: prefer r that hits more integers in the current window
        
        # To make this fast, note that changing residue from r_old to r_new only affects
        # integers in the arithmetic progressions. But we're adding a new prime, so
        # all integers that are ≡ r (mod p) get +1 coverage.
        
        # Let's evaluate candidates more cleverly:
        # The key integers to care about are those near the frontier
        # Specifically, frontier+1, frontier+2, ..., frontier+K for some small K
        
        # For each candidate residue r, the integers hit are: r, r+p, r+2p, ...
        # We want to maximize the new frontier
        
        # Quick filter: if cov[frontier+1] == 0, we MUST hit frontier+1
        # So r must satisfy r ≡ (frontier+1) (mod p), i.e., r = (frontier+1) % p
        # If cov[frontier+1] >= 1, we can choose any r, but hitting frontier+1 is good
        
        # Let's try a small set of candidate residues:
        # 1. r = (frontier+1) % p  (hits the immediate next integer)
        # 2. r = (frontier+2) % p  (hits the one after)
        # 3. r = 0
        # 4. r = frontier % p
        # 5. A few more around the frontier
        
        candidates = set()
        for offset in range(0, 10):
            candidates.add((frontier + 1 + offset) % p)
        candidates.add(0)
        candidates.add(frontier % p)
        
        # Also add the "natural" choice: hit as many already-singly-covered integers as possible
        # Integers with cov == 1 that are ≡ r (mod p) would become doubly covered
        # This is hard to compute quickly, so let's just evaluate the candidates
        
        best_r = 0
        best_frontier = -1
        best_hits = -1
        
        for r in candidates:
            # Simulate adding this residue
            # Count how many integers in [frontier+1, frontier+20] get their coverage increased
            # and check if the frontier extends
            
            new_frontier = frontier
            # Check from frontier+1 upward
            # An integer x is in the new doubly-covered prefix if cov[x] + (1 if x%p==r else 0) >= 2
            # We need this for all x from 1 to new_frontier
            # Since cov[x] >= 2 for x <= frontier (by definition), we only need to check x > frontier
            
            # Find the new frontier
            for x in range(frontier + 1, min(frontier + 50, BOUND + 1)):
                hit = 1 if (x % p == r) else 0
                if cov[x] + hit < 2:
                    break
                new_frontier = x
            
            # Count total hits in the window for tie-breaking
            hits = 0
            for x in range(1, min(frontier + 30, BOUND + 1)):
                if x % p == r:
                    hits += 1
            
            if new_frontier > best_frontier or (new_frontier == best_frontier and hits > best_hits):
                best_frontier = new_frontier
                best_hits = hits
                best_r = r
        
        # Apply the best residue
        residues[i] = best_r
        for x in range(best_r, BOUND + 1, p):
            if x >= 1:
                cov[x] += 1
        
        # Update frontier (should be at least best_frontier)
        frontier = best_frontier
    
    return residues, frontier

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Run the greedy construction
    residues, frontier = _greedy_construct(primes, deadline)
    
    # If we have time, try to improve with local search
    if time.time() < deadline - 5:
        # Simple local search: try changing individual residues
        current_score = frontier
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, p in enumerate(primes):
                if time.time() > deadline - 2:
                    break
                old_r = residues[i]
                # Try a few alternative residues
                for offset in range(-5, 6):
                    new_r = (old_r + offset) % p
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    score = _compute_score(residues, primes, cap=500)
                    if score > current_score:
                        current_score = score
                        improved = True
                    else:
                        residues[i] = old_r
                        break
    
    return residues

# EVOLVE-BLOCK-END
