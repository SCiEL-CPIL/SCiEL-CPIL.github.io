import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We select all even numbers in [20, 200]
    # They are pairwise non-coprime (gcd >= 2)
    moduli = list(range(20, 201, 2))
    
    selected = []
    
    for m in moduli:
        if time.time() > deadline:
            break
            
        found = False
        for r in range(m):
            compatible = True
            for m_prev, r_prev in selected:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        
        # If no residue found, we skip this modulus.
        # This is unlikely for even moduli with this greedy approach.
        
    return selected
