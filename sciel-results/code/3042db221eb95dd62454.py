"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Greedy: larger moduli first
    divisors.sort(reverse=True)
    
    # cov_counts[i] = number of classes covering residue i
    cov_counts = np.zeros(base, dtype=np.int32)
    classes = []  # list of [m, r]
    
    # Greedy Phase
    for m in divisors:
        if time.time() > deadline - 10:
            classes.append([m, 0])
            cov_counts[0::m] += 1
            continue
            
        best_r = 0
        best_gain = -1
        
        # Find r that covers most currently uncovered residues
        for r in range(m):
            # Count how many 0s are in cov_counts[r::m]
            gain = np.count_nonzero(cov_counts[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
                
        classes.append([m, best_r])
        cov_counts[best_r::m] += 1
        
    # Local Search Phase
    # We iterate until no improvement or time runs out.
    # Using incremental updates to cov_counts.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        for i in range(len(classes)):
            if time.time() > deadline - 5:
                break
                
            m, r = classes[i]
            
            # Calculate loss: residues in current (m, r) that are covered ONLY by this class
            # These will become uncovered if we move.
            loss = np.count_nonzero(cov_counts[r::m] == 1)
            
            best_r = r
            max_gain = -1
            
            # Find r' that maximizes gain: residues in (m, r') that are currently uncovered
            # These will become covered.
            for r_prime in range(m):
                if r_prime == r:
                    continue
                gain = np.count_nonzero(cov_counts[r_prime::m] == 0)
                if gain > max_gain:
                    max_gain = gain
                    best_r = r_prime
            
            # Check if best_r is an improvement
            # Delta in uncovered count = loss - max_gain
            # We want Delta < 0, so max_gain > loss
            if best_r != r and max_gain > loss:
                # Apply change
                cov_counts[r::m] -= 1
                cov_counts[best_r::m] += 1
                classes[i][1] = best_r
                improved = True
    
    return classes
# EVOLVE-BLOCK-END
