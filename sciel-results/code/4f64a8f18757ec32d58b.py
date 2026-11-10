"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Process moduli in increasing order (highest weight 1/m first)
    selected = []  # list of [m, r]
    
    for m in range(20, 201):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Find a residue r for m that is compatible with all selected
        # Constraint: for each (m_j, r_j) in selected, r != r_j (mod gcd(m, m_j))
        found = False
        for r in range(m):
            compatible = True
            for (m_j, r_j) in selected:
                g = gcd(m, m_j)
                if r % g == r_j % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        
        # If no residue found, skip this modulus (shouldn't happen often)
        if not found:
            continue
    
    return selected
# EVOLVE-BLOCK-END
