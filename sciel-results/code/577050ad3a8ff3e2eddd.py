import numpy as np
import time
import random

def construct():
    # Generate primes up to 1000
    primes = []
    for num in range(2, 1001):
        if all(num % i for i in range(2, int(num**0.5) + 1)):
            primes.append(num)
    
    num_primes = len(primes)
    # Start with a reasonable L_max. If we saturate this, we could increase it,
    # but 2000 is a good target for "L >= 1000".
    L_max = 2000
    
    # Initialize residues to 0
    residues = [0] * num_primes
    
    # counts[i] corresponds to integer i+1
    counts = np.zeros(L_max, dtype=np.int32)
    
    # Helper to get start index for slicing
    def get_start(r, p):
        return r - 1 if r > 0 else p - 1

    # Initialize counts
    for i, p in enumerate(primes):
        start = get_start(0, p)
        counts[start : L_max : p] += 1
        
    def get_score(counts):
        # Find first index where count < 2
        # If none, return L_max
        # If found at j, return j (since 1..j are covered)
        bad = np.nonzero(counts < 2)[0]
        if len(bad) == 0:
            return L_max
        return bad[0]
        
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues.copy()
    
    start_time = time.time()
    timeout = 110.0
    
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Iterate through primes
        for i in range(num_primes):
            p = primes[i]
            old_r = residues[i]
            best_r_for_p = old_r
            best_score_for_p = current_score
            
            # Try all residues for this prime
            for new_r in range(p):
                if new_r == old_r:
                    continue
                
                # Apply change
                start_old = get_start(old_r, p)
                start_new = get_start(new_r, p)
                
                counts[start_old : L_max : p] -= 1
                counts[start_new : L_max : p] += 1
                
                new_score = get_score(counts)
                
                if new_score > best_score_for_p:
                    best_score_for_p = new_score
                    best_r_for_p = new_r
                
                # Revert
                counts[start_new : L_max : p] -= 1
                counts[start_old : L_max : p] += 1
                
            # Apply best found
            if best_r_for_p != old_r:
                start_old = get_start(old_r, p)
                start_new = get_start(best_r_for_p, p)
                
                counts[start_old : L_max : p] -= 1
                counts[start_new : L_max : p] += 1
                
                residues[i] = best_r_for_p
                current_score = best_score_for_p
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                    
                    # If we hit L_max, we might want to expand, but for now
                    # just continue to refine.
        
        if not improved:
            no_improve_count += 1
            # Perturb if stuck
            if no_improve_count > 3:
                for _ in range(5):
                    idx = random.randint(0, num_primes - 1)
                    p = primes[idx]
                    old_r = residues[idx]
                    new_r = random.randint(0, p - 1)
                    if new_r != old_r:
                        start_old = get_start(old_r, p)
                        start_new = get_start(new_r, p)
                        counts[start_old : L_max : p] -= 1
                        counts[start_new : L_max : p] += 1
                        residues[idx] = new_r
                        current_score = get_score(counts)
                no_improve_count = 0
    
    return best_residues
