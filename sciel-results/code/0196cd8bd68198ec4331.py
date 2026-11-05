import time
import math
import numpy as np
import random

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
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    primes = _primes(N + 1)
    
    # Initialize residues
    # Greedy init: pick residue with min count in current set
    # Or just random
    residues = {}
    mask = np.ones(N + 1, dtype=bool)
    
    # Greedy initialization
    for p in primes:
        if time.time() > deadline - 5:
            break
        counts = np.bincount(np.where(mask)[0] % p, minlength=p)
        # If all counts are 0, it's empty, stop
        if counts.sum() == 0:
            break
        # Pick residue with min count
        r = np.argmin(counts)
        residues[p] = r
        mask[r::p] = False
        
    # Current best
    best_mask = mask.copy()
    best_size = mask.sum()
    best_residues = residues.copy()
    
    # Simulated Annealing
    # We optimize residues for ALL primes in `primes`
    # Note: If we restrict to p <= N, we might remove too much.
    # But the condition is only for p <= |A|.
    # However, if we satisfy for ALL p, it's valid.
    # Is it optimal to satisfy for all p?
    # Yes, because any valid A must satisfy for p <= |A|.
    # If we satisfy for p > |A| as well, we might reduce |A| unnecessarily.
    # So we should ONLY enforce constraints for p <= |A|.
    # This brings us back to the dynamic problem.
    
    # Hybrid approach:
    # 1. Get a candidate set A from greedy. Size K.
    # 2. Optimize residues for p <= K.
    # 3. Check if new size K' > K.
    # 4. If yes, update K and include more primes.
    # 5. If no, try to add elements?
    
    # Let's stick to the "Static Constraints for p <= N" first to see if it yields a good baseline.
    # Actually, satisfying p > K is wasteful.
    # We should only constrain p <= K.
    
    # Revised Plan:
    # 1. Greedy init to get K0.
    # 2. Local Search on residues for p <= K0.
    #    - If we change residue for p <= K0, recalc A.
    #    - If size increases, K increases.
    #    - If K increases, we must add constraints for new primes p in (K0, K_new].
    #      For these new primes, we must pick residues that don't reduce size too much.
    #      We can pick the residue with min count in the new A.
    
    # This is complex. Let's try a simpler structural change:
    # Randomized Greedy with Multiple Restarts.
    # Run the greedy algorithm many times with random tie-breaking or random initial residue choices.
    # Keep the best result.
    
    # This is a valid structural change (from deterministic to stochastic search).
    
    best_size = -1
    best_A = []
    
    num_restarts = 50 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        A = list(range(N + 1))
        # Shuffle primes? No, order matters for greedy.
        # But we can randomize the choice of residue if there are ties, or even randomly among top 3.
        
        p_idx = 0
        p = primes[0]
        
        # Randomize initial set? No, start full.
        # But we can randomize the residue choice.
        
        while p_idx < len(primes) and p <= len(A):
            if time.time() > deadline - 10:
                break
                
            # Count residues
            counts = [0] * p
            for x in A:
                counts[x % p] += 1
                
            if max(counts) == 0:
                break
                
            # Find min count
            min_count = min(counts)
            # Get all residues with min count
            candidates = [r for r in range(p) if counts[r] == min_count]
            
            # Randomly pick one
            r = random.choice(candidates)
            
            # Remove
            A = [x for x in A if x % p != r]
            
            p_idx += 1
            if p_idx < len(primes):
                p = primes[p_idx]
                
        if len(A) > best_size:
            best_size = len(A)
            best_A = A
            
    return best_A
