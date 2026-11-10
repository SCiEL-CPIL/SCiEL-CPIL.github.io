import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety

    selected = []  # List of (m, r)
    
    # Iterate through potential moduli from smallest to largest
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Try to find a residue r for m that is compatible with all selected
        # r must satisfy: for each (m_i, r_i) in selected, r != r_i mod gcd(m, m_i)
        found = False
        for r in range(m):
            compatible = True
            for (m_i, r_i) in selected:
                g = math.gcd(m, m_i)
                if r % g == r_i % g:
                    compatible = False
                    break
            if compatible:
                selected.append((m, r))
                found = True
                break
        # If no residue found, skip this modulus
    
    return [[m, r] for (m, r) in selected]
# EVOLVE-BLOCK-END
