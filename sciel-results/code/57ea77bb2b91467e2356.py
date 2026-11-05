"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Precompute positions for each (modulus, residue) pair
    # For each modulus m, positions[r] = list of positions congruent to r mod m
    positions_cache = {}
    for m in divisors:
        positions_cache[m] = [list(range(r, base, m)) for r in range(m)]
    
    def count_uncovered(covered):
        return int(np.sum(covered == 0))
    
    # Greedy initialization with a twist: process in a smart order
    # Try multiple orderings and keep the best
    best_residues = None
    best_uncovered = base  # worst case
    
    orderings = [
        list(range(n_div)),           # increasing modulus
        list(range(n_div-1, -1, -1)), # decreasing modulus
        sorted(range(n_div), key=lambda i: divisors[i], reverse=True),
        sorted(range(n_div), key=lambda i: base // divisors[i]),  # by density
    ]
    
    for order in orderings:
        if time.time() > deadline - 60:
            break
        
        residues = [0] * n_div
        covered = np.zeros(base, dtype=np.uint8)
        
        for idx in order:
            if time.time() > deadline - 60:
                break
            m = divisors[idx]
            best_r = 0
            best_count = -1
            for r in range(m):
                positions = positions_cache[m][r]
                count = int(np.sum(covered[positions] == 0))
                if count > best_count:
                    best_count = count
                    best_r = r
            residues[idx] = best_r
            covered[best_r::m] = 1
        
        current_uncovered = count_uncovered(covered)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]
    
    # Now do local search from the best initial solution
    residues = best_residues[:]
    covered = np.zeros(base, dtype=np.uint8)
    for i, m in enumerate(divisors):
        covered[residues[i]::m] = 1
    current_uncovered = count_uncovered(covered)
    
    # Local search with simulated annealing
    temp = 50.0
    cooling = 0.995
    
    # Precompute for faster delta computation
    # For each modulus, precompute the position arrays as numpy arrays
    pos_arrays = {}
    for m in divisors:
        pos_arrays[m] = [np.array(positions_cache[m][r]) for r in range(m)]
    
    while time.time() < deadline - 10:
        if temp < 0.5:
            break
        
        # Pick a random modulus to modify
        i = random.randint(0, n_div - 1)
        mi = divisors[i]
        old_ri = residues[i]
        
        # Pick new residue
        new_ri = random.randint(0, mi - 1)
        if new_ri == old_ri:
            continue
        
        # Compute delta efficiently using the covered array
        # Remove old coverage, add new coverage, count change
        old_covered = covered.copy()
        
        # Zero out old positions
        covered[old_ri::mi] = 0
        # Add new positions
        covered[new_ri::mi] = 1
        
        new_uncovered = count_uncovered(covered)
        delta = new_uncovered - current_uncovered
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            residues[i] = new_ri
            current_uncovered = new_uncovered
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
        else:
            covered[:] = old_covered
        
        temp *= cooling
    
    # Final local search pass: try all residues for each modulus
    # (hill climbing)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n_div):
            if time.time() > deadline - 5:
                break
            m = divisors[i]
            old_r = residues[i]
            best_r = old_r
            best_local = current_uncovered
            
            for r in range(m):
                if r == old_r:
                    continue
                old_covered = covered.copy()
                covered[old_r::m] = 0
                covered[r::m] = 1
                local_uncovered = count_uncovered(covered)
                if local_uncovered < best_local:
                    best_local = local_uncovered
                    best_r = r
                covered[:] = old_covered
            
            if best_r != old_r:
                covered[old_r::m] = 0
                covered[best_r::m] = 1
                current_uncovered = best_local
                residues[i] = best_r
                improved = True
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues[:]
    
    # Return the best solution found
    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
