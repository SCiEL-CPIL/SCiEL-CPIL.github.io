import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 120s limit

    # Generate primes up to 1000
    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)
    p_list = primes

    def compute_coverage(residues, limit):
        """Compute coverage for x in 1..limit."""
        cov = np.zeros(limit + 1, dtype=np.int16)
        for i, p in enumerate(p_list):
            r = int(residues[i])
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                cov[start::p] += 1
        return cov

    def score_from_coverage(cov, limit):
        """Find largest L such that cov[x] >= 2 for all x in 1..L."""
        bad_indices = np.where(cov[1:limit+1] < 2)[0]
        if len(bad_indices) == 0:
            return limit
        return int(bad_indices[0])

    def full_score(residues, limit=100000):
        cov = compute_coverage(residues, limit)
        return score_from_coverage(cov, limit)

    # Phase 1: Greedy construction
    # For each prime, choose the residue that maximizes coverage of the prefix
    # We work with a moderate limit for speed during construction
    CONSTRUCT_LIMIT = 20000
    
    # Initialize residues greedily
    residues = np.zeros(n_primes, dtype=np.int32)
    cov = np.zeros(CONSTRUCT_LIMIT + 1, dtype=np.int16)
    
    for i in range(n_primes):
        p = p_list[i]
        best_r = 0
        best_score = -1
        
        # Try all possible residues for this prime
        # But we can be smart: focus on residues that cover early numbers
        # Actually, let's try a subset first for speed, then refine
        # For small primes, try all; for large primes, try a subset
        if p <= 100:
            candidates = range(p)
        else:
            # Try residues that cover numbers in the current bottleneck region
            # Find current bottleneck
            bad = np.where(cov[1:CONSTRUCT_LIMIT+1] < 2)[0]
            if len(bad) == 0:
                # No bottleneck, try random candidates
                candidates = [random.randint(0, p-1) for _ in range(20)]
            else:
                bottleneck_x = int(bad[0]) + 1  # first bad number
                # Residues that cover bottleneck_x
                # x ≡ r (mod p) => r = x mod p
                r_bottleneck = bottleneck_x % p
                candidates = [r_bottleneck]
                # Add some random candidates
                for _ in range(15):
                    candidates.append(random.randint(0, p-1))
        
        for r in candidates:
            # Compute marginal benefit: how many currently singly-covered numbers
            # would become doubly covered, minus how many doubly-covered would become triply
            # Simplified: count how many numbers in 1..CONSTRUCT_LIMIT that are currently
            # covered exactly once and would be hit by this residue
            if r == 0:
                start = p
            else:
                start = r
            if start > CONSTRUCT_LIMIT:
                benefit = 0
            else:
                positions = np.arange(start, CONSTRUCT_LIMIT+1, p)
                # Count how many are currently covered exactly once
                benefit = np.sum(cov[positions] == 1)
                # Penalize if we're adding to already-doubly-covered (waste)
                waste = np.sum(cov[positions] >= 2)
                net_benefit = benefit - 0.1 * waste
            
            if net_benefit > best_score:
                best_score = net_benefit
                best_r = r
        
        residues[i] = best_r
        # Update coverage
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= CONSTRUCT_LIMIT:
            cov[start::p] += 1
    
    # Recompute coverage fresh
    cov = compute_coverage(residues, CONSTRUCT_LIMIT)
    best_score = score_from_coverage(cov, CONSTRUCT_LIMIT)
    best_residues = residues.copy()

    # Phase 2: Bottleneck repair
    # Repeatedly find the first bottleneck and fix it
    REPAIR_LIMIT = 50000
    
    for iteration in range(200):  # max repair iterations
        if time.time() > deadline - 20:
            break
        
        # Get current coverage up to REPAIR_LIMIT
        cov = compute_coverage(best_residues, REPAIR_LIMIT)
        current_score = score_from_coverage(cov, REPAIR_LIMIT)
        
        if current_score >= REPAIR_LIMIT:
            break
        
        # Find first bottleneck
        bad_indices = np.where(cov[1:REPAIR_LIMIT+1] < 2)[0]
        if len(bad_indices) == 0:
            break
        
        bottleneck_x = int(bad_indices[0]) + 1  # first number with cov < 2
        
        # Find which primes currently cover bottleneck_x
        covering_primes = []
        non_covering_primes = []
        for i, p in enumerate(p_list):
            r = int(best_residues[i])
            if (bottleneck_x % p) == r:
                covering_primes.append(i)
            else:
                non_covering_primes.append(i)
        
        # We need to change a non-covering prime to cover bottleneck_x
        # Choose the prime that, when changed to cover bottleneck_x,
        # maximizes the number of other singly-covered numbers it hits
        
        best_prime_idx = -1
        best_new_r = -1
        best_gain = -1
        
        # Try a subset of non-covering primes for speed
        if len(non_covering_primes) > 30:
            # Prefer smaller primes (they cover more numbers in the range)
            non_covering_primes.sort(key=lambda i: p_list[i])
            candidates_primes = non_covering_primes[:30]
        else:
            candidates_primes = non_covering_primes
        
        for i in candidates_primes:
            p = p_list[i]
            old_r = int(best_residues[i])
            new_r = bottleneck_x % p  # residue that covers bottleneck_x
            
            if new_r == old_r:
                continue
            
            # Compute gain: how many currently singly-covered numbers
            # in 1..REPAIR_LIMIT would become doubly covered
            if new_r == 0:
                new_start = p
            else:
                new_start = new_r
            if new_start > REPAIR_LIMIT:
                continue
            
            new_positions = np.arange(new_start, REPAIR_LIMIT+1, p)
            # Count how many are currently covered exactly once
            gain = np.sum(cov[new_positions] == 1)
            
            # Also consider: does this hurt anything? 
            # Old residue covered some numbers; those lose one coverage
            if old_r == 0:
                old_start = p
            else:
                old_start = old_r
            if old_start <= REPAIR_LIMIT:
                old_positions = np.arange(old_start, REPAIR_LIMIT+1, p)
                # Numbers that were covered exactly once and lose that coverage
                loss = np.sum(cov[old_positions] == 1)
            else:
                loss = 0
            
            net_gain = gain - loss
            
            if net_gain > best_gain:
                best_gain = net_gain
                best_prime_idx = i
                best_new_r = new_r
        
        if best_prime_idx >= 0:
            best_residues[best_prime_idx] = best_new_r
        else:
            # No good move found; try random perturbation
            i = random.randint(0, n_primes - 1)
            p = p_list[i]
            best_residues[i] = random.randint(0, p - 1)
        
        # Check if we improved
        new_cov = compute_coverage(best_residues, REPAIR_LIMIT)
        new_score = score_from_coverage(new_cov, REPAIR_LIMIT)
        
        if new_score <= current_score:
            # Revert if no improvement
            # We don't track old value easily, so just continue
            pass
    
    # Final evaluation with full limit
    final_score = full_score(best_residues, 100000)
    
    return best_residues.tolist()
