# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave buffer)

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
    primes_list = primes

    # Evaluation bound
    EVAL_BOUND = 10000

    def compute_counts(residues, bound):
        """Compute coverage counts for positions 1..bound."""
        counts = np.zeros(bound + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= bound:
                idx = np.arange(first, bound + 1, p)
                counts[idx] += 1
        return counts

    def get_L_from_counts(counts, bound):
        """Find largest L such that counts[1..L] >= 2."""
        # We only care up to bound
        below = np.where(counts[1:bound+1] < 2)[0]
        if len(below) == 0:
            return bound
        return int(below[0])

    def evaluate(residues):
        # Start with a smaller bound, expand if necessary? 
        # For speed in search, we can evaluate up to a dynamic bound.
        # But for greedy initialization, we need a fixed target.
        bound = EVAL_BOUND
        counts = compute_counts(residues, bound)
        return get_L_from_counts(counts, bound)

    def evaluate_local(residues, current_L):
        """Evaluate up to current_L + 1000 to see if we can extend."""
        bound = min(current_L + 1000, EVAL_BOUND)
        counts = compute_counts(residues, bound)
        return get_L_from_counts(counts, bound)

    random.seed(42)
    
    best_L = 0
    best_residues = [0] * n_primes

    # Phase 1: Greedy Construction
    # Target a specific L for initialization, e.g., 1000
    target_L = 1000
    
    # We will iterate a few times with different prime orderings
    for restart in range(10):
        if time.time() > deadline - 20.0:
            break
            
        # Shuffle prime order for diversity in greedy steps
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        current_residues = [0] * n_primes
        # Initialize counts for the target range
        target_bound = target_L
        counts = np.zeros(target_bound + 1, dtype=np.int32)
        
        # Greedy step: For each prime, pick residue that maximizes "benefit"
        # Benefit = number of positions x in [1, target_L] where x % p == r 
        # AND counts[x] < 2.
        
        for idx in prime_indices:
            p = primes_list[idx]
            best_r = 0
            best_score = -1
            
            # Try all residues? p can be up to 997.
            # We can optimize by only checking residues that appear in [1, target_L]
            # or just iterate 0..p-1.
            # Since p <= 1000, iterating p times is fast.
            
            # Precompute which positions are covered by which residues for this p?
            # Or just loop r from 0 to p-1 and count.
            # For speed, we can compute the counts for each residue class in the target range.
            
            # positions covered by residue r: r, r+p, r+2p...
            # We want to maximize sum(1 for x in covered if counts[x] < 2)
            
            # Efficiently:
            # Create an array of "need" where need[x] = 1 if counts[x] < 2 else 0
            need = (counts < 2).astype(np.int32)
            
            max_score = -1
            best_r = 0
            
            for r in range(p):
                if r == 0:
                    first = p
                else:
                    first = r
                if first > target_bound:
                    continue
                
                # Get indices
                idxs = np.arange(first, target_bound + 1, p)
                # Sum need at these indices
                score = np.sum(need[idxs])
                
                if score > max_score:
                    max_score = score
                    best_r = r
            
            current_residues[idx] = best_r
            
            # Update counts
            r = best_r
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= target_bound:
                idxs = np.arange(first, target_bound + 1, p)
                counts[idxs] += 1
        
        # Evaluate the greedy result
        current_L = evaluate(current_residues)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current_residues[:]

    # Phase 2: Local Search (Hill Climbing)
    # Start from best greedy solution
    current = best_residues[:]
    current_L = best_L
    
    no_improve = 0
    max_no_improve = 20
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        if current_L >= EVAL_BOUND:
            break
            
        # Find the first gap
        gap_pos = current_L + 1
        
        # Try to fix the gap by changing one prime's residue
        # Candidate primes: those that do NOT cover gap_pos
        candidates = []
        for i, p in enumerate(primes_list):
            if current[i] != gap_pos % p:
                candidates.append(i)
        
        if not candidates:
            break
            
        random.shuffle(candidates)
        
        improved = False
        # Try a subset of candidates
        for i in candidates[:50]:
            if time.time() > deadline - 5.0:
                break
                
            p = primes_list[i]
            old_r = current[i]
            new_r = gap_pos % p
            
            if new_r == old_r:
                continue
                
            current[i] = new_r
            new_L = evaluate_local(current, current_L)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
                break
            else:
                current[i] = old_r
        
        if not improved:
            no_improve += 1
            # If stuck, try a random perturbation to escape local minima
            if no_improve > 5:
                # Perturb 5 random residues
                for _ in range(5):
                    idx = random.randint(0, n_primes - 1)
                    p = primes_list[idx]
                    current[idx] = random.randint(0, p - 1)
                current_L = evaluate(current)
                if current_L < best_L:
                    # Revert to best if perturbation was bad
                    current = best_residues[:]
                    current_L = best_L
                no_improve = 0

    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
