"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    residues = [0] * n_primes
    
    # Helper to calculate L for a given set of residues
    # Optimization: We only need to check up to a certain limit.
    # L is the largest integer such that all 1..L have coverage >= 2.
    # We can estimate an upper bound. If we have 168 primes, max coverage at any x is 168.
    # But we need coverage >= 2.
    # A safe upper bound for L is small initially, but can grow.
    # Let's check up to 5000 initially, expand if needed.
    
    def get_L(res, max_check=10000):
        # Count coverage for 1..max_check
        # cov[x] = sum(1 for i, p in enumerate(primes) if x % p == res[i])
        # We want the first x where cov[x] < 2.
        # Efficient way:
        # Initialize count array of 0s of size max_check+1
        counts = [0] * (max_check + 1)
        for i, p in enumerate(primes):
            r = res[i]
            # Mark all x in 1..max_check such that x % p == r
            # x = r, r+p, r+2p, ...
            # Note: x must be >= 1.
            # If r == 0, x = p, 2p, ...
            # If r > 0, x = r, r+p, ...
            
            # Start of sequence
            if r == 0:
                start = p
            else:
                start = r
                if start == 0: # r < p, so r=0 handled above. r>0 means start=r.
                    start = p # This shouldn't happen if r<p and r!=0
            # Wait, if r=5, p=10, x=5, 15, 25...
            # If r=0, p=10, x=10, 20...
            
            for x in range(start, max_check + 1, p):
                counts[x] += 1
        
        # Find first x in 1..max_check with counts[x] < 2
        # If all have >= 2, return max_check (or higher, but we capped)
        for x in range(1, max_check + 1):
            if counts[x] < 2:
                return x - 1
        return max_check

    # Strategy:
    # 1. Initialize residues to cover small numbers densely.
    #    Let's try to cover 1, 2, 3... with at least 2 primes each.
    #    A simple heuristic: assign residues randomly, then hill climb.
    #    Better: Greedy construction.
    #    Let's start with a random assignment and hill climb.
    
    random.seed(42)
    
    # Initial random assignment
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
    
    # Evaluate initial L
    current_L = get_L(residues, 2000)
    best_L = current_L
    best_residues = residues[:]
    
    # Hill Climbing
    # We want to increase L.
    # If current L is 100, the bottleneck is at 101 (coverage < 2).
    # We should try to increase coverage at 101.
    # Or, we might break a previously good number if we change a residue.
    # So, we only accept changes that increase L, or keep L same but improve "robustness"?
    # Simplest: Only accept if L increases.
    
    # To speed up, we can track which primes affect the current bottleneck.
    # But get_L is O(N_primes * L/p) which is fast enough for L < 10000.
    # 168 primes. Sum(10000/p) ~ 10000 * log(log(1000)) ~ 10000 * 2 ~ 20000 ops per eval.
    # 120s / (20000 ops) ~ 6000 evals. That's quite few for hill climbing.
    
    # We need a faster evaluation.
    # Incremental update:
    # If we change residue of prime i from old_r to new_r:
    # For each x in 1..max_check:
    #   if x % p_i == old_r: counts[x] -= 1
    #   if x % p_i == new_r: counts[x] += 1
    # Then find new L.
    
    # Let's maintain counts array and update it incrementally.
    
    def init_counts(res, max_check):
        counts = [0] * (max_check + 1)
        for i, p in enumerate(primes):
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, max_check + 1, p):
                counts[x] += 1
        return counts

    def update_counts(counts, i, old_r, new_r, max_check):
        p = primes[i]
        # Remove old
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        for x in range(start_old, max_check + 1, p):
            counts[x] -= 1
            
        # Add new
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        for x in range(start_new, max_check + 1, p):
            counts[x] += 1

    def find_L(counts, max_check):
        for x in range(1, max_check + 1):
            if counts[x] < 2:
                return x - 1
        return max_check

    # Set max_check to 10000
    max_check = 10000
    counts = init_counts(residues, max_check)
    current_L = find_L(counts, max_check)
    best_L = current_L
    best_residues = residues[:]
    
    # Hill Climbing Loop
    # We iterate through primes, try changing their residue.
    # To avoid getting stuck in local minima, we can use simulated annealing or random restarts.
    # Given time constraint, let's do a few rounds of random hill climbing.
    
    # We want to push L higher.
    # If L is low, the bottleneck is at L+1.
    # We should focus on primes that could cover L+1.
    # Primes p where (L+1) % p is a possible residue.
    # Actually, any prime can cover L+1 if we set r_p = (L+1) % p.
    
    # Let's prioritize changing residues of primes that currently do NOT cover the bottleneck,
    # to see if we can fix it without breaking others.
    
    # Optimization: If we change a residue, we must re-scan from 1 to find new L.
    # This is O(max_check).
    
    # Let's try a different approach:
    # Greedy assignment:
    # For each prime p (in increasing order of p? or decreasing?), assign r_p to maximize coverage of 1..K.
    
    # Let's stick to hill climbing with incremental updates.
    # It's simple and effective if we have enough time.
    
    # To improve speed, we can limit the search space for new residues.
    # Instead of trying all p residues, try a few random ones, or the one that covers the bottleneck.
    
    # Strategy:
    # 1. Identify bottleneck B = current_L + 1.
    # 2. For each prime i:
    #    a. Try setting r_i = B % p_i.
    #    b. Update counts.
    #    c. Find new L.
    #    d. If new_L > current_L: accept, update best.
    #    e. Else: revert.
    # 3. Repeat until no improvement or time out.
    # 4. Add random perturbations to escape local minima.
    
    # Let's implement this.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_L = find_L(counts, max_check)
        bottleneck = current_L + 1
        
        # If bottleneck > max_check, we've hit the cap.
        if bottleneck > max_check:
            break
            
        # Try to fix the bottleneck
        # We can try all primes, or a random subset.
        # Let's try all primes, but in random order to avoid bias.
        indices = list(range(n_primes))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            p = primes[i]
            old_r = residues[i]
            new_r = bottleneck % p
            
            if new_r == old_r:
                continue
                
            # Update counts
            update_counts(counts, i, old_r, new_r, max_check)
            residues[i] = new_r
            
            new_L = find_L(counts, max_check)
            
            if new_L > current_L:
                current_L = new_L
                best_L = new_L
                best_residues = residues[:]
                improved = True
                # If we improved, we should restart the loop or continue?
                # Continuing might find more improvements in the same pass.
                # But the bottleneck has changed.
                # Let's break and restart the outer loop with the new bottleneck.
                break
            else:
                # Revert
                update_counts(counts, i, new_r, old_r, max_check)
                residues[i] = old_r
        
    # If we still have time, do some random hill climbing to escape local minima
    # or try to push L further by looking at a slightly larger window?
    # No, L is defined by the first failure.
    
    # Let's try a few random restarts if time permits?
    # Or just return the best found.
    
    # One more pass: try to improve L by changing residues that affect the current bottleneck,
    # but also consider that changing one might break an earlier number.
    # The hill climbing above handles this by checking new_L.
    
    # If best_L is still low, maybe we need a better initial state?
    # The random start might be bad.
    # Let's try a heuristic start:
    # Assign r_p such that small numbers are covered.
    # For x in 1..100, we want at least 2 primes to cover it.
    
    # Given the time, the hill climbing from random start might be enough if it converges.
    # However, if it gets stuck at L=84 (seed score), we need to be sure it can escape.
    # The seed score is 1.0 (L=84).
    # The inspiration says hill climbing achieved 2.8 (L~234).
    # So it should work.
    
    # Return best residues
    return best_residues
# EVOLVE-BLOCK-END
