import numpy as np
import time
import random

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

# Precompute primes up to a reasonable limit, e.g., 600
PRIMES = _primes(600)

def construct():
    start_time = time.time()
    deadline = start_time + 320  # 10s margin
    
    # We want to find the largest admissible tuple.
    # An admissible tuple S of size k is a set of k distinct integers in [0, 5000]
    # such that for every prime p <= k, S does not cover all residue classes mod p.
    
    # This is equivalent to:
    # There exist residues r_p for each prime p <= k such that
    # S is a subset of {x in [0, 5000] : for all p <= k, x % p != r_p}.
    # And |S| >= k.
    
    # So we want to find the largest k such that there exist residues r_p for p <= k
    # where the number of x in [0, 5000] not congruent to r_p mod p for any p <= k is >= k.
    
    # Let's iterate k downwards from a high bound.
    # What is the max possible k?
    # The density of survivors is prod_{p<=k} (1 - 1/p).
    # By Mertens' theorem, prod_{p<=k} (1 - 1/p) ~ e^{-gamma} / ln(k).
    # So expected survivors ~ 5000 * e^{-gamma} / ln(k).
    # We need 5000 * e^{-gamma} / ln(k) >= k.
    # e^{-gamma} ~ 0.561.
    # 2805 / ln(k) >= k.
    # For k=100, ln(100)~4.6, 2805/4.6 ~ 600 >= 100.
    # For k=200, ln(200)~5.3, 2805/5.3 ~ 529 >= 200.
    # For k=300, ln(300)~5.7, 2805/5.7 ~ 492 >= 300.
    # For k=400, ln(400)~6.0, 2805/6.0 ~ 467 >= 400.
    # For k=500, ln(500)~6.2, 2805/6.2 ~ 452 < 500.
    # So max k is likely around 400-450.
    
    # We will search k from 450 down to 1.
    # For each k, we try to find residues r_p that maximize the count of survivors.
    # If max survivors >= k, we return a valid tuple of size k.
    
    # To find good residues, we can use a local search or greedy method.
    # Greedy: start with random residues, then iteratively change one residue to maximize survivors.
    
    # Let's implement a function to compute survivors count for given residues.
    
    def compute_survivors(residues, primes_k):
        # residues: dict p -> r
        # primes_k: list of primes <= k
        # Returns boolean array of survivors in [0, N]
        survivors = np.ones(N, dtype=bool)
        for p in primes_k:
            r = residues[p]
            survivors &= (np.arange(N) % p != r)
        return survivors
    
    # We will use a local search to find residues that maximize survivors.
    # Start with random residues.
    # Iterate: for each p, try all r, pick the one that maximizes survivors.
    # Repeat until no improvement.
    
    # To speed up, we can precompute X % p for all p? No, N is 5000, p is small.
    # X % p is fast.
    
    # Let's try this for each k.
    
    for k in range(450, 0, -1):
        if time.time() > deadline:
            break
            
        primes_k = [p for p in PRIMES if p <= k]
        if not primes_k:
            continue
            
        # Try to find residues that yield >= k survivors.
        # Use multiple random restarts to escape local optima.
        
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize residues randomly
            residues = {p: random.randint(0, p - 1) for p in primes_k}
            
            # Compute initial survivors
            survivors = compute_survivors(residues, primes_k)
            count = np.sum(survivors)
            
            # Local search
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                    
                improved = False
                for p in primes_k:
                    current_r = residues[p]
                    best_r = current_r
                    best_count = count
                    
                    for r in range(p):
                        if r == current_r:
                            continue
                        residues[p] = r
                        new_survivors = compute_survivors(residues, primes_k)
                        new_count = np.sum(new_survivors)
                        if new_count > best_count:
                            best_count = new_count
                            best_r = r
                    residues[p] = best_r
                    if best_r != current_r:
                        count = best_count
                        survivors = compute_survivors(residues, primes_k)
                        improved = True
            
            if count >= k:
                # Found a valid set of residues
                # Extract k distinct integers from survivors
                valid_indices = np.where(survivors)[0]
                # Shuffle and take first k
                np.random.shuffle(valid_indices)
                result = valid_indices[:k].tolist()
                return result
        
    # Fallback
    return [0]
# EVOLVE-BLOCK-END
