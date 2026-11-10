import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # List of all moduli in [20, 200]
    # Process in increasing order of m (highest weight 1/m first)
    selected = []
    
    for m in range(20, 201):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Try to find a valid residue r for m
        # We try residues in order 0, 1, ..., m-1
        # To improve chances, we could randomize, but deterministic is safer for reproducibility
        found = False
        for r in range(m):
            # Check compatibility with all selected (m', r')
            compatible = True
            for m_prime, r_prime in selected:
                g = gcd(m, m_prime)
                if r % g == r_prime % g:
                    compatible = False
                    break
            
            if compatible:
                selected.append([m, r])
                found = True
                break
        
        # If no residue found, m is skipped
    
    return selected
