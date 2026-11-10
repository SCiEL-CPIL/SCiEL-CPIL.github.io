"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Initial random solution
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    # Helper to compute score: largest L such that 1..L are all covered >= 2 times
    # We cap the search for L at 10^5 as per contract, but we can stop early if we find a break.
    # To speed up, we precompute for each prime p the set of covered numbers in a window?
    # No, 10^5 is small enough that a direct count is fast if vectorized.
    
    def compute_score(res):
        # We want the max L such that for all x in 1..L, count(x) >= 2
        # Let's check up to a reasonable limit. The seed got 84. 
        # Inspiration 1 got ~1.14, which might be L~95 or so.
        # Let's check up to 1000 first, then extend if needed.
        # Actually, the contract says score is L.
        
        # Efficient check:
        # For each prime p_i with residue r_i, the numbers covered are r_i, r_i+p_i, r_i+2p_i...
        # We can create a boolean array or a count array for 1..MAX_L.
        # MAX_L = 100000 is the cap, but we likely won't reach that.
        # Let's try to find the first x where count < 2.
        
        # Optimization: 
        # If we just want the prefix length, we can iterate x from 1 upwards and stop at first failure.
        # But checking counts for each x is O(n_primes).
        # Better: Precompute which primes cover which x? 
        # Or: Use the fact that we only need to check until we find a gap.
        
        # Let's use a numpy trick for a fixed window, say 2000.
        # If L > 2000, we extend.
        
        max_check = 5000
        counts = np.zeros(max_check + 1, dtype=np.int32)
        
        for p, r in zip(primes, res):
            # r is in [0, p-1]. 
            # Numbers are r, r+p, r+2p...
            # Note: if r==0, numbers are p, 2p... (since x>=1, 0 is not counted in 1..L)
            # Wait, the congruence is x = r (mod p).
            # If r=0, x is multiple of p.
            # If r>0, x is r, r+p...
            
            # Start index in 1..max_check
            # Smallest x >= 1 such that x % p == r
            if r == 0:
                first = p
            else:
                first = r
                if first < 1:
                    first += p # r is 0..p-1, so if r>0, r>=1. If r=0, first=p.
            
            if first <= max_check:
                # Add 1 to counts at first, first+p, ...
                indices = np.arange(first, max_check + 1, p)
                counts[indices] += 1
        
        # Find the first index i in 1..max_check where counts[i] < 2
        # We want the largest L such that 1..L all have count >= 2.
        # So we look for the first i where counts[i] < 2.
        # L = i - 1.
        
        # If all 1..max_check are >= 2, we return max_check (or check further).
        # But for now, let's assume L < 5000.
        
        # np.where(counts[1:] < 2)[0] gives indices in 0..max_check-1 corresponding to 1..max_check
        fails = np.where(counts[1:] < 2)[0]
        if len(fails) == 0:
            return max_check # Might need to extend, but 5000 is likely safe for now
        else:
            return fails[0] # This is the index in 1..max_check where it fails. 
                            # If fails[0]=0, it means counts[1]<2, so L=0.
                            # If fails[0]=5, it means counts[6]<2, so L=5.
                            # Wait. counts is 0-indexed.
                            # counts[1] corresponds to x=1.
                            # fails[0] is the index in the slice counts[1:].
                            # So if fails[0]=0, counts[1] < 2. L=0.
                            # If fails[0]=4, counts[5] < 2. L=4.
                            # So L = fails[0].
                            
        # Correction:
        # x=1 is counts[1].
        # If counts[1] < 2, fails[0] = 0. L should be 0.
        # If counts[1]>=2, counts[2]<2, fails[0]=1. L should be 1.
        # So L = fails[0].
        
        return int(fails[0])

    best_res = residues.copy()
    best_score = compute_score(best_res)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    # Move: Pick a random block of K residues and change them randomly.
    # K can vary.
    
    while time.time() < deadline and T > min_T:
        # Propose a move
        K = random.randint(1, 5) # Change 1 to 5 residues
        indices = random.sample(range(n_primes), K)
        
        new_res = best_res.copy()
        for idx in indices:
            p = primes[idx]
            new_res[idx] = random.randint(0, p - 1)
        
        new_score = compute_score(new_res)
        
        # Acceptance probability
        delta = new_score - best_score
        if delta > 0:
            best_res = new_res
            best_score = new_score
        else:
            # Accept with probability exp(delta / T)
            if random.random() < np.exp(delta / T):
                best_res = new_res
                best_score = new_score
        
        T *= cooling_rate
        
        # Occasionally, try a larger jump or restart if stuck?
        # For now, keep it simple.
        
        # Time check inside loop for safety
        if time.time() > deadline - 5:
            break
            
    return list(best_res)
# EVOLVE-BLOCK-END
