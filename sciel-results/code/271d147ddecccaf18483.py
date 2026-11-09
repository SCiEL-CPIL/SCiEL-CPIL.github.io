import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Generate primes <= 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Max L to check. The task caps at 10^5, but realistically we won't get there.
    # Let's check up to 2000 to be safe and fast.
    MAX_L_CHECK = 2000
    
    # Initialize residues with the seed strategy
    # Seed: ((i + 2) // 2) % p
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Initialize coverage counts
    # counts[x] = number of primes p such that x == r_p (mod p)
    counts = np.zeros(MAX_L_CHECK + 1, dtype=np.int32)
    
    # Apply initial residues to counts
    for i, p in enumerate(primes):
        r = residues[i]
        # Update counts for x = r, r+p, r+2p, ... <= MAX_L_CHECK
        # Using numpy slicing for speed
        counts[r:MAX_L_CHECK+1:p] += 1
        
    # Function to get the current L
    def get_L():
        # Find the first x >= 1 such that counts[x] < 2
        # If no such x, L is MAX_L_CHECK
        # Note: We only care about x in 1..MAX_L_CHECK
        bad_indices = np.where(counts[1:] < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L_CHECK
        else:
            return bad_indices[0]
            
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Main search loop
    stuck_counter = 0
    max_stuck = 1000
    
    while time.time() - start_time < time_limit:
        # Determine strategy
        if stuck_counter >= max_stuck:
            # Random kick to escape local optimum
            # Pick a random prime
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            new_r = np.random.randint(p)
            
            # Apply change
            counts[old_r:MAX_L_CHECK+1:p] -= 1
            residues[i] = new_r
            counts[new_r:MAX_L_CHECK+1:p] += 1
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                stuck_counter = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # Revert
                residues[i] = old_r
                counts[new_r:MAX_L_CHECK+1:p] -= 1
                counts[old_r:MAX_L_CHECK+1:p] += 1
                stuck_counter += 1
        else:
            # Greedy repair: Target the first failing integer
            # Current L is current_L. The first failing integer is current_L + 1
            # Wait, get_L returns the first x where count < 2.
            # So if current_L is 100, it means 1..100 are good, 101 is bad.
            # We want to fix 101.
            x_fail = current_L + 1
            
            if x_fail > MAX_L_CHECK:
                # We hit the check limit. We can try to expand MAX_L_CHECK or just stop.
                # For now, break or continue with random kicks.
                # Let's just do random kicks if we hit the limit.
                i = np.random.randint(n_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                counts[old_r:MAX_L_CHECK+1:p] -= 1
                residues[i] = new_r
                counts[new_r:MAX_L_CHECK+1:p] += 1
                
                new_L = get_L()
                
                if new_L > current_L:
                    current_L = new_L
                    stuck_counter = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    residues[i] = old_r
                    counts[new_r:MAX_L_CHECK+1:p] -= 1
                    counts[old_r:MAX_L_CHECK+1:p] += 1
                    stuck_counter += 1
                continue
            
            # Find primes that do NOT currently cover x_fail
            # p covers x_fail if residues[i] == x_fail % p
            # We want to pick a p where residues[i] != x_fail % p
            # And change it to cover x_fail.
            
            # To speed up, we can just pick a random prime and check if it covers x_fail.
            # If it does, we can't improve x_fail by changing it to cover x_fail (it already does).
            # We could change it to something else, but that might break other things.
            # So we only consider primes that DON'T cover x_fail.
            
            # Optimization: Precompute which primes cover x_fail?
            # x_fail changes.
            # Just loop through a few random primes.
            
            found_prime = False
            for _ in range(10): # Try 10 random primes
                i = np.random.randint(n_primes)
                p = primes[i]
                target_r = x_fail % p
                if residues[i] != target_r:
                    # This prime does not cover x_fail. We can make it cover x_fail.
                    found_prime = True
                    break
            
            if not found_prime:
                # All tried primes already cover x_fail? Unlikely.
                # Or we just didn't find one. Fallback to random kick.
                stuck_counter += 1
                continue
                
            # Apply the greedy move
            old_r = residues[i]
            new_r = target_r
            
            counts[old_r:MAX_L_CHECK+1:p] -= 1
            residues[i] = new_r
            counts[new_r:MAX_L_CHECK+1:p] += 1
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                stuck_counter = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # Revert
                residues[i] = old_r
                counts[new_r:MAX_L_CHECK+1:p] -= 1
                counts[old_r:MAX_L_CHECK+1:p] += 1
                stuck_counter += 1

    return best_residues

# EVOLVE-BLOCK-END
