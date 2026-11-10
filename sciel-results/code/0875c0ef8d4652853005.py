"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Greedy strategy: Process moduli in increasing order of m (highest weight 1/m first)
    # For each m, pick the smallest residue r that is compatible with all previous selections.
    selected = []
    
    for m in range(20, 201):
        # Check deadline
        if time.time() > deadline:
            break
            
        # Try residues in order 0, 1, ..., m-1
        # Note: The specific choice of r (smallest vs largest) might affect future conflicts.
        # Smallest r is a common heuristic.
        found = False
        for r in range(m):
            compatible = True
            for m2, r2 in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        # If no residue found for this m, skip it (should not happen for small m with sparse conflicts)
        
    return selected
# EVOLVE-BLOCK-END
