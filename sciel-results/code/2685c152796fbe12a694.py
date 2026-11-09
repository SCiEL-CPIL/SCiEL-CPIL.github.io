import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate primes >= 13 and <= 1000
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(13, 1001) if is_prime(p)]
    n_primes = len(primes)
    
    # Max L to check. The cap is 10^5, but likely we won't reach that.
    # Let's set a reasonable upper bound for the search space, e.g., 5000.
    # If we reach 5000, we can expand.
    max_L_check = 5000
    
    # Initial residues: same as seed
    # i-th prime covers i+1
    residues = [i + 1 for i in range(n_primes)]
    
    # count[x] = number of primes covering x
    # We need to handle x up to max_L_check
    counts = [0] * (max_L_check + 1)
    
    def update_counts(p, r, delta):
        # delta is +1 or -1
        # Update counts for x = r, r+p, r+2p ... <= max_L_check
        # Note: r is in 0..p-1. But we cover 1..L.
        # If r == 0, it covers p, 2p, ...
        # If r > 0, it covers r, r+p, ...
        # The class is [p, r]. x is covered if x % p == r.
        # For x in 1..max_L_check:
        # if r == 0:
        #   x = p, 2p, ...
        # else:
        #   x = r, r+p, ...
        
        if r == 0:
            start_x = p
        else:
            start_x = r
            
        for x in range(start_x, max_L_check + 1, p):
            counts[x] += delta

    # Initialize counts
    for i, p in enumerate(primes):
        r = residues[i]
        update_counts(p, r, 1)
        
    def get_L():
        for x in range(1, max_L_check + 1):
            if counts[x] == 0:
                return x - 1
        return max_L_check

    best_L = get_L()
    best_residues = residues.copy()
    
    # Local Search
    # We try to improve L by changing residues.
    # Strategy: Find first gap x. Try to cover it by changing some prime's residue.
    
    while time.time() < deadline:
        current_L = get_L()
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        # If we reached the max check bound, we might want to expand, 
        # but for now let's assume 5000 is enough or we stop if we hit it.
        if best_L >= max_L_check:
            break
            
        gap_x = current_L + 1
        
        # Try to cover gap_x
        improved = False
        # Randomize order of primes to avoid bias
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        for idx in prime_indices:
            p = primes[idx]
            old_r = residues[idx]
            new_r = gap_x % p
            
            if old_r == new_r:
                continue
                
            # Apply change
            update_counts(p, old_r, -1)
            residues[idx] = new_r
            update_counts(p, new_r, 1)
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                # We found an improvement. We can break or continue.
                # Breaking might be faster to find the next gap.
                # But maybe another change improves more?
                # Let's break to handle the new gap immediately.
                break
            else:
                # Revert
                update_counts(p, new_r, -1)
                residues[idx] = old_r
                update_counts(p, old_r, 1)
                
        if not improved:
            # Stuck. Try a random kick?
            # Pick a random prime and change its residue randomly.
            if random.random() < 0.1: # 10% chance to kick
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                
                update_counts(p, old_r, -1)
                residues[idx] = new_r
                update_counts(p, new_r, 1)
                
                # If this kick results in a worse L, it might still help escape a local min
                # in a subsequent step. But we should track best.
                # For now, we just let it be.
                
    return [[primes[i], best_residues[i]] for i in range(n_primes)]
