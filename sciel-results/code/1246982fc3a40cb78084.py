import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Determine valid residues for modulus m
        valid = [True] * m
        
        for (m_i, r_i) in selected:
            d = math.gcd(m, m_i)
            r_i_mod_d = r_i % d
            # Mark all r in [0, m) where r mod d == r_i_mod_d as invalid
            for r in range(r_i_mod_d, m, d):
                valid[r] = False
        
        # Find the smallest valid residue
        r_chosen = None
        for r in range(m):
            if valid[r]:
                r_chosen = r
                break
        
        if r_chosen is not None:
            selected.append((m, r_chosen))
    
    return [[m, r] for m, r in selected]
