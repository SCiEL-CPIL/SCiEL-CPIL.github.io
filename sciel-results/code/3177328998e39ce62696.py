"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    n = len(primes)
    # We want to find the largest L where min count >= 2 for all x in 1..L
    # Use a counting approach: for each prime, mark which numbers in 1..max_check it hits
    # But that's O(n * max_check) which is 168 * 100000 = 16.8M, feasible
    
    # Actually, let's be smarter. We need the minimum count over x=1..L.
    # Let's compute counts for x=1..max_check
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # x ≡ r (mod p), so x = r, r+p, r+2p, ...
        # But x must be >= 1, so first x is max(r, 1) if r>0 else p
        # Actually x = r + k*p for k>=0, and x >= 1
        # If r == 0, x = p, 2p, 3p, ...
        # If r > 0, x = r, r+p, r+2p, ...
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    
    # Find the largest L such that min(counts[1:L+1]) >= 2
    # We can find the first x where counts[x] < 2
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Deadline guard
    deadline = time.time() + 140  # 140s out of 150s
    
    # Start with the seed solution
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _score(best_residues, primes)
    
    # Try local search: perturb one residue at a time
    # For each prime index, try changing its residue to a nearby value
    # and see if the score improves
    
    # We'll use a greedy hill-climbing approach
    # For efficiency, we can compute the score incrementally, but for now
    # let's do full score computation and limit iterations
    
    # Number of iterations we can afford
    # Each score computation is O(n * max_check / p_avg) ≈ 168 * 100000 / 500 ≈ 33600 ops
    # So we can do maybe 1000-5000 iterations in 140s
    
    # Let's do a more targeted search: for each position, try a few candidate residues
    # and keep the best
    
    improved = True
    max_rounds = 100
    round_num = 0
    
    while improved and round_num < max_rounds and time.time() < deadline:
        improved = False
        round_num += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Try a few candidates: current, and some neighbors
            # Focus on residues that might help extend the coverage
            # Key insight: we want each integer 1..L to be hit by at least 2 primes
            # So we want to distribute residues to cover gaps
            
            # Try current and a few random/strategic alternatives
            candidates = [current_r]
            # Add some strategic candidates
            for delta in [-1, 1, -2, 2, -3, 3]:
                new_r = (current_r + delta) % p
                if new_r not in candidates:
                    candidates.append(new_r)
            
            # Also try a few random candidates
            if p > 6:
                for _ in range(3):
                    new_r = np.random.randint(0, p)
                    if new_r not in candidates:
                        candidates.append(new_r)
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                test_residues = list(best_residues)
                test_residues[i] = new_r
                
                s = _score(test_residues, primes)
                
                if s > best_score:
                    best_residues = test_residues
                    best_score = s
                    improved = True
                    break  # Move to next i
    
    return best_residues
# EVOLVE-BLOCK-END
