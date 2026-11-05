import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        # Find a valid residue for this modulus
        valid_r = None
        for r in range(m):
            compatible = True
            for m_old, r_old in selected:
                g = math.gcd(m, m_old)
                if r % g == r_old % g:
                    compatible = False
                    break
            if compatible:
                valid_r = r
                break
        if valid_r is not None:
            selected.append((m, valid_r))
    
    return [[m, r] for m, r in selected]
