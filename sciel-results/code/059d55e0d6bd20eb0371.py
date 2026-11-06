import math
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Greedy selection: iterate moduli in increasing order (smallest m first)
    # For each m, try to find a valid residue r that is disjoint from all selected
    # If found, add it. This maximizes count of high-weight 1/m terms.
    
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Find a valid residue r in [0, m) such that for all (m2, r2) in selected:
        # r % gcd(m, m2) != r2 % gcd(m, m2)
        
        # Precompute gcds with all selected moduli
        gcds = []
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            gcds.append((g, r2 % g))
        
        # Try residues 0, 1, 2, ... and pick the first valid one
        # Since we just need ANY valid residue, try in order
        found = False
        for r in range(m):
            valid = True
            for g, r2_mod in gcds:
                if r % g == r2_mod:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                found = True
                break
        
        # If no valid residue found for this m, skip it
    
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in selected]
