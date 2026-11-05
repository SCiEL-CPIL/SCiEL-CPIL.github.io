"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import random

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
    
    # We will optimize all primes
    # Initialize residues
    # Heuristic: Start with r_p = 1 for all, or random
    residues = [1] * n_primes
    
    # Max L to check during search
    # The problem asks for L >= 1000. Let's aim a bit higher to be safe, say 5000.
    # If we reach 5000, the score is at least 1000.
    MAX_L = 5000
    
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
        # np.argmax returns the index of the first True value
        # We want the first index where counts < 2
        # If all are >= 2, argmax returns 0 (since no True? No, argmax on empty/False returns 0? 
        # Actually, np.argmax(np.array([False, False])) returns 0.
        # Better: use np.where
        # But for speed, a loop is okay if L is small, but numpy is better.
        # Let's use a simple loop for clarity and sufficient speed for L ~ 1000-5000
        # 5000 iterations is negligible.
        for x in range(1, MAX_L + 1):
            if counts[x] < 2:
                return x - 1
        return MAX_L

    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    max_iterations = 100000  # Adjust based on time
    
    random.seed(42)
    
    for i in range(max_iterations):
        # Pick a random prime index
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        
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
        
        # Acceptance Criteria
        if new_L >= current_L:
            # Accept improvement or equal
            current_L = new_L
            residues[idx] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
        else:
            # Worse solution
            # Accept with probability exp(-delta / T)
            delta = current_L - new_L
            if delta < T:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            
            if random.random() < prob:
                current_L = new_L
                residues[idx] = new_r
            else:
                # Revert
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                # residues[idx] remains old_r

        # Cool down
        T *= cooling_rate
        if T < T_min:
            # Reheat to escape local minima? Or just stop?
            # Reheating is part of Iterated Simulated Annealing
            # Let's do a simple restart if T gets too low and we haven't improved in a while
            # For simplicity, just keep cooling. 100k iterations with 0.999 cooling
            # T starts at 10, ends at 10 * 0.999^100000 ~ 10 * e^-100 ~ 0.
            # This is effectively a long anneal.
            pass

    return best_residues

# EVOLVE-BLOCK-END
