"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _calculate_score(residues, primes, max_check=10000):
    """
    Calculate the largest L such that every integer 1..L satisfies at least two congruences.
    Uses vectorized counting for speed.
    """
    # We only need to check up to a reasonable limit. 
    # The theoretical max is bounded, but 10000 is a safe upper bound for search efficiency 
    # given the 120s constraint. If L > max_check, we return max_check.
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for r, p in zip(residues, primes):
        # x = r (mod p)
        # First positive integer satisfying this:
        # If r == 0, it's p, 2p, ...
        # If r > 0, it's r, r+p, ...
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
    # We need counts[x] >= 2 for all x in 1..L
    # Find the first index where count < 2
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    else:
        # below_two[0] is the 0-indexed offset in counts[1:], so the actual integer is below_two[0] + 1
        # The largest L that is fully covered is (below_two[0] + 1) - 1 = below_two[0]
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Initial Heuristic: 
    # Try to assign residues such that small numbers are covered.
    # A simple greedy initialization: assign r_p such that p divides a small number?
    # Actually, let's start with a random initialization or a simple pattern, then optimize.
    # Pattern from seed: ((i + 2) // 2) % p
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    current_score = _calculate_score(residues, primes)
    
    # Local Search: Hill Climbing
    # We will try to change one residue at a time.
    # To speed up, we can pre-compute the "impact" of changes, but simple re-evaluation
    # with NumPy is fast enough for 168 primes if we limit iterations.
    
    # Optimization: 
    # Changing r_p affects counts at positions r_p, r_p+p, etc.
    # Instead of full re-scan, we can update counts incrementally.
    
    # Let's implement incremental update for speed.
    max_check = 10000
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(res, prim, delta):
        # delta is +1 or -1
        start = res if res > 0 else prim
        if start <= max_check:
            counts[start::prim] += delta

    # Initialize counts
    for r, p in zip(residues, primes):
        update_counts(r, p, 1)
        
    # Helper to get score from counts quickly
    def get_score_from_counts():
        # Check 1..max_check
        # We can maintain a running variable or just check the prefix.
        # For simplicity in the loop, we'll just check the first failure point.
        # But doing np.where every time is O(N). 
        # Let's try to be smarter: keep track of the current L.
        # Actually, just recalculating the "first failure" is fast enough in C via NumPy if we limit max_check.
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    best_score = current_score
    best_residues = residues.copy()
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # Iterate through primes in random order or fixed order?
        # Random order helps escape local minima.
        idxs = np.random.permutation(n_primes)
        
        for i in idxs:
            if time.time() > deadline:
                break
                
            p = primes[i]
            old_r = residues[i]
            
            # Try a few random new residues for this prime
            # How many to try? Maybe 5-10 random ones.
            for _ in range(10):
                new_r = np.random.randint(0, p)
                if new_r == old_r:
                    continue
                
                # Apply change
                update_counts(old_r, p, -1)
                residues[i] = new_r
                update_counts(new_r, p, 1)
                
                new_score = get_score_from_counts()
                
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues.copy()
                    improved = True
                    no_improve_count = 0
                    # Accept and keep this change
                    break
                else:
                    # Revert
                    update_counts(new_r, p, -1)
                    residues[i] = old_r
                    update_counts(old_r, p, 1)
            
            if improved:
                # If we found an improvement, restart the outer loop? 
                # Or just continue? Continuing is fine, but resetting no_improve is key.
                # We break out of the prime loop to restart the permutation? 
                # Actually, just continue to the next prime.
                pass
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
