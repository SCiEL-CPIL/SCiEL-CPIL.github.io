import time
import math

def construct():
    # Deadline guard, though the algorithm is very fast
    start_time = time.time()
    
    selected = []  # List of [m, r]
    
    # Iterate through moduli in increasing order to prioritize smaller m (larger 1/m)
    for m in range(20, 201):
        # Try to find a valid residue r for this modulus m
        for r in range(m):
            compatible = True
            # Check compatibility with all previously selected classes
            for m_i, r_i in selected:
                g = math.gcd(m, m_i)
                if r % g == r_i % g:
                    compatible = False
                    break
            
            if compatible:
                selected.append([m, r])
                break
        
        # Safety margin check, though unlikely to trigger
        if time.time() - start_time > 110:
            break
    
    return selected
