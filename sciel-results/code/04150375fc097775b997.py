"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    covered = bytearray(base)
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
        
        for r in range(m):
            # Count uncovered residues in the arithmetic progression r, r+m, r+2m, ...
            count = covered[r::m].count(0)
            if count > best_count:
                best_count = count
                best_r = r
        
        classes.append([m, best_r])
        # Mark covered positions
        covered[best_r::m] = b'\x01' * len(range(best_r, base, m))
    
    return classes
# EVOLVE-BLOCK-END
