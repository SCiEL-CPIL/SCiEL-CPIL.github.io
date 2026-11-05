"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
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
    
    # We will optimize the first K primes
    K = 20
    small_primes = primes[:K]
    
    # Initialize residues
    # Heuristic: Start with r_p = 1 for all
    residues = [1] * n_primes
    
    # Max L to check during search
    MAX_L = 2000
    
    # Initialize counts array
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start == 0:
            start = p
        if start <= MAX_L:
            counts[start::p] += 1
            
    def get_L(counts):
        # Find first x in 1..MAX_L such that counts[x] < 2
        # Using a Python loop is fast enough if L is not huge
        # If L is huge, we might want to use numpy, but L is likely < 1000 initially
        for x in range(1, MAX_L + 1):
            if counts[x] < 2:
                return x - 1
        return MAX_L

    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    # Local Search
    random.seed(42) # For reproducibility in evaluation if needed, or just random
    
    # Number of iterations
    # We have 120s limit. 10000 iterations is very fast.
    # Let's do 50000 iterations to be safe and thorough.
    for _ in range(50000):
        # Pick a random small prime index
        idx = random.randint(0, K - 1)
        p = small_primes[idx]
        
        # Pick a random new residue
        new_r = random.randint(0, p - 1)
        old_r = residues[idx]
        
        if new_r == old_r:
            continue
            
        # Update counts: Remove old_r coverage
        start_old = old_r
        if start_old == 0:
            start_old = p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        # Update counts: Add new_r coverage
        start_new = new_r
        if start_new == 0:
            start_new = p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        # Evaluate new L
        new_L = get_L(counts)
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues[:]
            current_L = new_L
            # Accept change
            residues[idx] = new_r
        else:
            # Revert change
            if start_new <= MAX_L:
                counts[start_new::p] -= 1
            if start_old <= MAX_L:
                counts[start_old::p] += 1
            # residues[idx] stays old_r
            
        # Optional: If we are stuck for a while, do a random restart of small primes?
        # Or accept equal moves with low probability?
        # Let's add a small chance to accept equal moves to escape local minima
        if new_L == best_L and random.random() < 0.05:
            residues[idx] = new_r
            # Note: counts are already updated to new_r state in this branch?
            # No, in the else branch we reverted counts.
            # So if we accept equal, we need to keep the counts in new_r state.
            # The logic above:
            # If new_L > best_L: counts are in new_r state.
            # If new_L <= best_L: counts are reverted to old_r state.
            # So for the "accept equal" case, we need to NOT revert.
            
            # Let's restructure the accept/revert logic
            pass # The code above handles >. We need to handle == carefully.

    # Let's rewrite the loop logic to be cleaner
    # Actually, the previous logic is fine if we just don't revert in the equal case.
    
    # Re-implementing loop for clarity
    residues = [1] * n_primes
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start == 0:
            start = p
        if start <= MAX_L:
            counts[start::p] += 1
            
    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    for _ in range(50000):
        idx = random.randint(0, K - 1)
        p = small_primes[idx]
        new_r = random.randint(0, p - 1)
        old_r = residues[idx]
        
        if new_r == old_r:
            continue
            
        start_old = old_r
        if start_old == 0: start_old = p
        start_new = new_r
        if start_new == 0: start_new = p
        
        # Apply new_r
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        new_L = get_L(counts)
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues[:]
            residues[idx] = new_r
        elif new_L == best_L and random.random() < 0.1:
            # Accept to explore
            residues[idx] = new_r
        else:
            # Revert
            if start_new <= MAX_L:
                counts[start_new::p] -= 1
            if start_old <= MAX_L:
                counts[start_old::p] += 1
            # residues[idx] remains old_r

    # Final check with larger MAX_L
    # Use best_residues to compute final score
    # The evaluator will do this, but we can ensure we return a valid list
    
    return best_residues

# EVOLVE-BLOCK-END
