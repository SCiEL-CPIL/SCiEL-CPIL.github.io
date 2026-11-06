import math
import time

def construct():
    """
    Constructs a set of pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Returns:
        list of [m, r] pairs
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    selected = []
    
    # We want to include the smallest moduli first to maximize sum(1/m)
    # Iterate through moduli from 20 to 200
    for m in range(20, 201):
        if time.time() > deadline:
            break
            
        # Try to find a valid residue r for this modulus m
        found = False
        for r in range(m):
            # Check compatibility with all previously selected moduli
            compatible = True
            for m_prime, r_prime in selected:
                g = math.gcd(m, m_prime)
                if r % g == r_prime % g:
                    compatible = False
                    break
            
            if compatible:
                selected.append([m, r])
                found = True
                break
        
        # If no residue found, we skip this modulus.
        # Given the structure of GCDs, it's rare to skip small moduli,
        # but for larger moduli with many common factors, it might happen.
        # However, since we want to maximize sum(1/m), skipping a small m is bad,
        # but if impossible, we must. The greedy approach ensures we keep the
        # smallest possible moduli that allow a valid configuration.
        
    return selected
