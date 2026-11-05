"""Optimized solver for the covering problem."""
import random
import time
import numpy as np
from collections import defaultdict

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _count_covered(residues, primes, max_L):
    """
    Count how many integers in 1..max_L are covered at least twice.
    Returns the largest L such that every integer 1..L is covered at least twice.
    """
    # For each prime p_i with residue r_i, the covered positions are r_i, r_i+p, r_i+2p, ...
    # We need to find the largest L such that for all x in 1..L, count(x) >= 2.
    
    # Efficient approach: use an array to track coverage counts, but that's O(L * num_primes) which is too slow for large L.
    # Instead, we can use the fact that we only need to check up to some bound.
    # Actually, let's just compute the coverage for each x in 1..max_L using a difference array or direct counting.
    
    # For efficiency with numpy:
    # For each residue r_i mod p_i, the positions are r_i + k*p_i for k=0,1,...
    # We want to know for each x in 1..max_L, how many i satisfy x ≡ r_i (mod p_i).
    
    # Direct approach: for each x, check all primes. This is O(max_L * 168) which for max_L=1000 is ~168k ops, fine.
    # For max_L=100000, it's ~16.8M ops, also fine in numpy if vectorized.
    
    # Let's use a vectorized approach:
    # Create a matrix where row i is the residues for prime i broadcast over 1..max_L
    # This would be 168 * max_L memory, which for max_L=100000 is 16.8M entries, ~134MB for int32. Might be tight.
    
    # Alternative: iterate over primes and accumulate counts.
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # Positions: r, r+p, r+2p, ... up to max_L
        # But r could be 0, in which case positions are 0, p, 2p, ...
        # We want 1..max_L, so if r==0, start at p (since 0 is not in 1..max_L unless we consider it)
        # Actually x ≡ 0 (mod p) means x is a multiple of p.
        # Let's handle this: the positions in 1..max_L are r + k*p where r+k*p >= 1.
        # If r == 0: positions are p, 2p, 3p, ...
        # If r > 0: positions are r, r+p, r+2p, ...
        
        if r == 0:
            start = p
        else:
            start = r
        
        if start > max_L:
            continue
            
        # Generate positions
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    
    # Find the largest L such that counts[1..L] >= 2
    # We need min(counts[1..L]) >= 2 for all up to L
    # Find the first position where counts < 2
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Initial solution from the seed
    initial_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_residues = initial_residues[:]
    best_L = _count_covered(best_residues, primes, 100000)
    
    # Local search: try to improve by changing individual residues
    # For each prime, try different residues to see if it improves the score
    
    # We'll do iterative improvement: repeatedly find the best single-residue change
    # and apply it until no improvement is found or time runs out.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 5:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Try all other residues for this prime
            best_new_r = current_r
            best_new_L = best_L
            
            # We can sample residues if p is large, but for p <= 1000, trying all is feasible
            # However, _count_covered is O(168 * L) which for L=100000 is ~16.8M per call
            # Trying all p residues for each of 168 primes would be 168 * avg(p) * 16.8M which is too slow
            # avg(p) for primes up to 1000 is ~500, so 168 * 500 = 84000 calls * 16.8M = way too slow
            
            # We need a faster evaluation. Let's use a smarter approach.
            # When we change residue for prime i, only the coverage of that prime changes.
            # So we can compute the delta in coverage.
            
            # Let's precompute the coverage counts for the current solution
            # And then for each candidate residue, compute the new coverage efficiently
            
            # Actually, let's just do a random search with a time limit
            # Try random residues for each prime and see if it improves
            
            # Better: hill climbing with random restarts
            # For now, let's just try a few random changes
            
            # Try a random subset of residues for this prime
            num_trials = min(p, 50)  # Cap trials per prime
            candidates = set()
            if p <= 100:
                candidates = set(range(p)) - {current_r}
            else:
                for _ in range(num_trials):
                    candidates.add(random.randint(0, p - 1))
                candidates.discard(current_r)
            
            for new_r in candidates:
                if (time.time() - start_time) > deadline - 5:
                    break
                test_residues = best_residues[:]
                test_residues[i] = new_r
                L = _count_covered(test_residues, primes, 100000)
                if L > best_new_L:
                    best_new_L = L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                best_residues[i] = best_new_r
                best_L = best_new_L
                improved = True
    
    return best_residues
