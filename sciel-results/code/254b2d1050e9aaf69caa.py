# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

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
    primes_list = primes # Keep as Python list for fast modulo

    # Evaluation bound: we want to maximize L, cap at 10000 for safety/speed
    EVAL_BOUND = 10000

    def compute_counts(residues):
        """Compute coverage counts for positions 1..EVAL_BOUND."""
        counts = np.zeros(EVAL_BOUND + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= EVAL_BOUND:
                # Use slicing for speed
                # Indices: first, first+p, ...
                # np.arange is fine
                idx = np.arange(first, EVAL_BOUND + 1, p)
                counts[idx] += 1
        return counts

    def get_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        # We need the first index i in 1..EVAL_BOUND where counts[i] < 2
        # counts[0] is unused
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return EVAL_BOUND
        # below[0] is 0-indexed relative to counts[1:], so actual index is below[0] + 1
        # The number of fully covered integers is (below[0] + 1) - 1 = below[0]
        return int(below[0])

    def evaluate(residues):
        counts = compute_counts(residues)
        return get_L_from_counts(counts)

    # Initial best
    best_L = 0
    best_residues = [0] * n_primes

    # Random restarts with Greedy Repair
    random.seed(42)
    
    while time.time() < deadline:
        # Initialize current residues
        # Strategy: Start with random, or perturb best
        if best_L > 0 and random.random() < 0.5:
            current = best_residues[:]
            # Perturb 10% of residues
            for _ in range(n_primes // 10):
                idx = random.randint(0, n_primes - 1)
                p = primes_list[idx]
                current[idx] = random.randint(0, p - 1)
        else:
            current = [random.randint(0, p - 1) for p in primes_list]
        
        current_L = evaluate(current)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
        
        # Greedy Repair Loop
        # Try to extend L by fixing the first gap
        max_repair_steps = 2000 # Limit repair steps per restart to prevent hanging
        steps = 0
        
        while current_L < EVAL_BOUND and steps < max_repair_steps:
            if time.time() > deadline - 2.0:
                break
            
            gap_pos = current_L + 1
            if gap_pos > EVAL_BOUND:
                break
            
            # Find primes that do NOT cover gap_pos
            # A prime p covers gap_pos if current[i] == gap_pos % p
            candidates = []
            for i, p in enumerate(primes_list):
                if current[i] != gap_pos % p:
                    candidates.append(i)
            
            if not candidates:
                # All primes cover gap_pos? Then gap_pos is not a gap.
                # This shouldn't happen if current_L is correct, unless counts were wrong.
                # Or maybe it is covered but count < 2? No, if all cover, count >= 1.
                # Wait, if all primes cover it, count is 168 >= 2. So it's not a gap.
                # So if we are here, it means my logic "current_L is first gap" is inconsistent?
                # No, get_L_from_counts finds first < 2.
                # If all primes cover it, count is 168. So it can't be the first < 2.
                # So candidates must not be empty?
                # Actually, if candidates is empty, it means ALL primes cover gap_pos.
                # Then count[gap_pos] >= 168 >= 2. So gap_pos is NOT a gap.
                # Contradiction. So this branch is likely unreachable or indicates a bug in L calc.
                # Just break to be safe.
                break

            # Shuffle candidates to try random ones first
            random.shuffle(candidates)
            
            improved = False
            # Try up to 20 candidates
            for i in candidates[:20]:
                p = primes_list[i]
                old_r = current[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue # Should not happen due to candidate filter

                current[i] = new_r
                new_L = evaluate(current)
                
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = current[:]
                    break # Restart the gap finding with new L
                else:
                    current[i] = old_r
            
            if not improved:
                # Stuck at this L for this configuration
                break
            
            steps += 1
            
            if best_L >= 1000:
                break

        if best_L >= 1000:
            break

    # Ensure Python integers
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
