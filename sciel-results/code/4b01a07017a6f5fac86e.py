"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Greedy: process moduli in ascending order (smaller m first for higher 1/m)
    # For each modulus, pick a residue compatible with all previously selected classes
    selected = []  # list of [m, r]
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Try each residue r in [0, m)
        found = False
        for r in range(m):
            # Check compatibility with all previously selected classes
            compatible = True
            for (m2, r2) in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        
        if not found:
            # No valid residue for this modulus; skip it
            # (Should be rare with ascending order)
            pass

    return selected
# EVOLVE-BLOCK-END
