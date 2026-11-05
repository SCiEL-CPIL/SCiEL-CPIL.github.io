"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Get all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute for each divisor m, the list of residues it covers
    # For speed, we'll use numpy arrays
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    # Greedy: process moduli in increasing order
    # For each modulus, pick the residue that covers the most uncovered residues
    for m in divisors:
        if time.time() > deadline:
            break
        
        # Find the best residue r for modulus m
        # We want to maximize the number of uncovered residues in {r, r+m, r+2m, ...} mod base
        best_r = 0
        best_count = -1
        
        # Use numpy to efficiently compute coverage counts for all residues at once
        # For each residue r in [0, m), count uncovered in r::m
        # This can be done by reshaping covered into a 2D array
        
        # covered[i] for i = r, r+m, r+2m, ... 
        # We can compute this by taking covered and slicing
        
        # Efficient approach: for each r, count = sum(covered[r::m] == 0)
        # But this is slow in pure Python loop. Let's use numpy.
        
        # Alternative: create a 2D view if possible, but m varies
        # For each m, we can compute all counts at once
        
        # Create indices: for residue r, the positions are r, r+m, r+2m, ...
        # Number of terms: (base - r) // m + 1 if r < base else 0
        
        # Vectorized approach:
        # For each r in 0..m-1, positions are r + k*m for k=0,1,...
        # We can use advanced indexing
        
        # Simpler: just loop over r but use numpy sum
        # This should be fast enough for m up to 51975
        
        for r in range(m):
            # Count uncovered residues in the arithmetic progression
            # positions: r, r+m, r+2m, ..., < base
            # Use numpy slicing
            mask = covered[r::m]
            count = np.count_nonzero(mask == 0)
            if count > best_count:
                best_count = count
                best_r = r
        
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    return classes
# EVOLVE-BLOCK-END
