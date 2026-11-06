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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We will use a greedy set-cover approach.
    # The goal is to maximize L such that every integer 1..L is covered by at least 2 residues.
    # This is equivalent to finding residues such that the "coverage count" is >= 2 for the longest prefix.
    
    # To make the greedy heuristic effective, we need to track which numbers are "bottlenecks".
    # A number x is a bottleneck if its current coverage count is 1.
    # When we assign a residue to a prime p, we want to cover as many bottleneck numbers as possible.
    
    # However, the "bottleneck" status changes as we assign more primes.
    # So we process primes one by one.
    # For prime p, we try all possible residues r in 0..p-1.
    # For each r, we calculate how many numbers in the range [1, current_max_check] 
    # that are currently covered by exactly 1 other prime would be covered by this new residue.
    # Actually, it's simpler: we want to maximize the "double coverage" extension.
    
    # Let's maintain an array `counts` of size MAX_CHECK + 1.
    # Initially all 0.
    # We process primes in increasing order.
    # For each prime p, we choose r_p to maximize the score.
    # But calculating the exact score (largest L) for every r_p is expensive (O(MAX_CHECK)).
    # Instead, we can use a heuristic: maximize the number of positions x in [1, MAX_CHECK] 
    # such that counts[x] == 1 and x is covered by r_p (mod p).
    # This directly helps extend the prefix of double-covered numbers.
    
    MAX_CHECK = 10000
    counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Precompute for each prime p, the indices it covers for each residue r?
    # No, that's too much memory.
    # We can just loop through r for each p.
    
    # Optimization: 
    # The "gain" of choosing residue r for prime p is the number of x in [1, MAX_CHECK] 
    # such that x = r (mod p) AND counts[x] == 1.
    # If we choose the r that maximizes this gain, we are effectively fixing the most "urgent" gaps.
    
    # Let's implement this.
    
    for i, p in enumerate(primes):
        if time.time() > deadline - 10:
            # Not enough time to finish properly, break and return best so far
            break
            
        best_r = 0
        best_gain = -1
        
        # We only care about x in [1, MAX_CHECK]
        # For a given r, the positions are r, r+p, r+2p, ... (adjusting for r=0 case: p, 2p...)
        
        # To speed up, we can precompute the "mask" of positions covered by p?
        # Or just iterate.
        # p can be up to 997. MAX_CHECK is 10000.
        # Number of positions per residue is ~10000/p.
        # Total work per prime p is sum over r of (10000/p) = 10000.
        # So total work is sum(10000) for all primes = 168 * 10000 = 1.68M operations.
        # This is very fast in Python if vectorized, or even in a tight loop.
        
        # Let's try to vectorize the gain calculation.
        # Create an array of all x in [1, MAX_CHECK] where counts[x] == 1.
        # single_covered = np.where(counts[1:] == 1)[0] + 1
        
        # For each r, we want to count how many x in single_covered satisfy x % p == r.
        
        # This is equivalent to:
        # For a fixed p, we have a set of numbers S (single covered).
        # We want to find r that maximizes |{x in S : x % p == r}|.
        
        # We can do this by:
        # 1. Get the residues of all single-covered numbers modulo p.
        # 2. Count the frequency of each residue.
        # 3. Pick the residue with the highest frequency.
        
        single_covered = np.where(counts[1:MAX_CHECK+1] == 1)[0] + 1
        
        if len(single_covered) > 0:
            # Calculate residues of single_covered numbers modulo p
            # Note: np.mod works element-wise
            mods = single_covered % p
            
            # Count occurrences of each residue
            # np.bincount is fast
            # We need to ensure the length is p
            counts_mod = np.bincount(mods, minlength=p)
            
            # Find the residue with max count
            best_r = np.argmax(counts_mod)
            best_gain = counts_mod[best_r]
        else:
            # No single covered numbers? 
            # This means we have already double covered everything up to MAX_CHECK?
            # Or we have 0 or 2+ covered everywhere.
            # If we have 0 covered somewhere, we can't fix it with this prime if it's not in the "gap".
            # But if counts[x] is 0, it's not in single_covered.
            # If counts[x] is >= 2, it's not in single_covered.
            # So if single_covered is empty, we don't have any "urgent" gaps to fix.
            # In this case, we can pick any residue, or one that helps future steps.
            # A good heuristic: pick r that covers the most 0-count numbers?
            # Or just 0.
            # Let's pick r to maximize coverage of 0-count numbers if possible, otherwise 0.
            zero_covered = np.where(counts[1:MAX_CHECK+1] == 0)[0] + 1
            if len(zero_covered) > 0:
                mods_zero = zero_covered % p
                counts_zero = np.bincount(mods_zero, minlength=p)
                best_r = np.argmax(counts_zero)
            else:
                best_r = 0
        
        residues[i] = best_r
        
        # Update counts
        r = best_r
        start = r if r > 0 else p
        if start <= MAX_CHECK:
            counts[start::p] += 1
    
    # Now we have a set of residues.
    # We can try to improve this with a local search (hill climbing) for the remaining time.
    
    current_score = 0
    # Calculate initial score
    below_two = np.where(counts[1:MAX_CHECK+1] < 2)[0]
    if len(below_two) == 0:
        current_score = MAX_CHECK
    else:
        current_score = below_two[0]
        
    best_score = current_score
    best_residues = residues.copy()
    
    # Local Search Phase
    # We will try to change one residue at a time to improve the score.
    # To speed up, we maintain the counts array and update incrementally.
    
    def update_counts(res, prim, delta):
        start = res if res > 0 else prim
        if start <= MAX_CHECK:
            counts[start::prim] += delta
            
    def get_score_from_counts():
        below_two = np.where(counts[1:MAX_CHECK+1] < 2)[0]
        if len(below_two) == 0:
            return MAX_CHECK
        return below_two[0]

    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        idxs = np.random.permutation(n_primes)
        
        for i in idxs:
            if time.time() > deadline - 5:
                break
                
            p = primes[i]
            old_r = residues[i]
            
            # Try a few random new residues
            # Also try the "greedy best" residue based on current single-covered set
            single_covered = np.where(counts[1:MAX_CHECK+1] == 1)[0] + 1
            candidate_rs = set()
            
            if len(single_covered) > 0:
                mods = single_covered % p
                counts_mod = np.bincount(mods, minlength=p)
                # Get top 5 residues
                top_rs = np.argsort(counts_mod)[-5:]
                candidate_rs.update(top_rs)
                
            # Add some random ones
            candidate_rs.update(np.random.randint(0, p, 10))
            
            for new_r in candidate_rs:
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
                    break
                else:
                    # Revert
                    update_counts(new_r, p, -1)
                    residues[i] = old_r
                    update_counts(old_r, p, 1)
            
            if improved:
                pass # Continue to next prime in permutation
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
