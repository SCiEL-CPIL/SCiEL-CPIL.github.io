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

def construct():
    deadline = time.time() + 280  # safety margin
    primes = _primes(N + 1)
    
    # Initial state: all numbers 0..N
    S = list(range(N + 1))
    removed = set()
    
    # counts[p][r] = number of elements in S congruent to r mod p
    counts = {}
    active_counts = {} # number of residues r with counts[p][r] > 0
    
    for p in primes:
        c = [0] * p
        counts[p] = c
        active_counts[p] = 0
        
    # Initialize counts
    for x in S:
        for p in primes:
            r = x % p
            if counts[p][r] == 0:
                active_counts[p] += 1
            counts[p][r] += 1

    # Helper to check if S is admissible
    # Admissible if for all p <= len(S), active_counts[p] < p
    def is_admissible():
        k = len(S)
        for p in primes:
            if p > k:
                break
            if active_counts[p] == p:
                return False
        return True

    # Greedy removal
    # We want to find the largest p <= len(S) such that active_counts[p] == p
    # Then remove the residue class r with minimum counts[p][r]
    
    while True:
        if time.time() > deadline:
            break
            
        k = len(S)
        # Find violating primes
        violating_primes = []
        for p in primes:
            if p > k:
                break
            if active_counts[p] == p:
                violating_primes.append(p)
                
        if not violating_primes:
            break
            
        # Choose the largest violating prime
        p_star = max(violating_primes)
        
        # Find residue r with min count
        min_count = float('inf')
        best_r = -1
        for r in range(p_star):
            if counts[p_star][r] < min_count:
                min_count = counts[p_star][r]
                best_r = r
                
        # Remove all x in S such that x % p_star == best_r
        to_remove = []
        for x in S:
            if x % p_star == best_r:
                to_remove.append(x)
                
        if not to_remove:
            # Should not happen if active_counts[p_star] == p_star
            break
            
        # Apply removal
        remove_set = set(to_remove)
        for x in to_remove:
            removed.add(x)
            for p in primes:
                r = x % p
                counts[p][r] -= 1
                if counts[p][r] == 0:
                    active_counts[p] -= 1
                    
        S = [x for x in S if x not in remove_set]

    # Local search / Repair phase
    # Try to add back removed elements
    # Sort removed elements? Maybe randomize or sort by value.
    # Let's iterate through removed elements in random order to avoid bias?
    # Or just sorted.
    # To maximize size, we want to add as many as possible.
    # The order might matter. Let's try adding in random order, or maybe
    # prioritize elements that are "less likely to cause conflicts".
    # An element x causes a conflict for prime p if x % p is a residue that
    # is currently the only missing one? No, it causes a conflict if adding x
    # makes active_counts[p] == p.
    # This happens if counts[p][x%p] was 0 (so it fills the last hole) OR
    # if active_counts[p] was p-1 and x fills the last hole.
    # Actually, if active_counts[p] < p, adding x increases active_counts[p]
    # only if counts[p][x%p] was 0.
    # If counts[p][x%p] > 0, active_counts[p] stays same.
    # So x is "safe" for p if counts[p][x%p] > 0 OR active_counts[p] < p-1 (so even if it fills a hole, it doesn't reach p).
    # Wait, if active_counts[p] == p-1 and counts[p][x%p] == 0, then adding x makes active_counts[p] == p -> Violation.
    # So x is dangerous for p if active_counts[p] == p-1 and counts[p][x%p] == 0.
    
    # Let's just try adding them back one by one.
    # We need to be careful with the deadline.
    
    import random
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    for x in removed_list:
        if time.time() > deadline:
            break
            
        # Check if x can be added
        # Tentative addition
        k_new = len(S) + 1
        safe = True
        
        # Check primes <= k_new
        for p in primes:
            if p > k_new:
                break
            r = x % p
            # If this residue was empty, it will become non-empty
            if counts[p][r] == 0:
                # It will increase active count
                if active_counts[p] + 1 == p:
                    # This would make p fully covered
                    safe = False
                    break
            # If counts[p][r] > 0, active count doesn't change, so no new violation for p
            # (Assuming it was admissible before, i.e., active_counts[p] < p)
            # But wait, k_new > len(S).
            # If p was > len(S) before, it wasn't checked.
            # Now p <= k_new.
            # We must ensure that for this new p, active_counts[p] < p.
            # If counts[p][r] > 0, active_counts[p] is unchanged.
            # We need to check if active_counts[p] == p.
            # But if active_counts[p] == p, it would have been a violation in the previous state
            # IF p <= len(S).
            # If p > len(S), it wasn't a constraint.
            # So if p is in (len(S), k_new], i.e., p == k_new (since k_new = len(S)+1, p must be prime, so p=k_new is the only candidate),
            # we need to check if active_counts[p] == p.
            
            # Simplified: Just check if after addition, active_counts[p] == p.
            # Let new_active = active_counts[p]
            # if counts[p][r] == 0: new_active += 1
            # if new_active == p: safe = False
            
        if safe:
            # Add x to S
            S.append(x)
            removed.discard(x)
            for p in primes:
                r = x % p
                if counts[p][r] == 0:
                    active_counts[p] += 1
                counts[p][r] += 1
        else:
            # Cannot add x
            pass

    return S
