import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_count = 0
    best_result = []
    
    moduli = list(range(2, 129))
    
    # We will try many random restarts
    # To increase diversity, we can also try different strategies for residue selection
    # e.g. random, or sequential with random offset
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        random.shuffle(moduli)
        
        current_pairs = []
        
        for m in moduli:
            # Try to find a residue r for m that is compatible with all current_pairs
            found = False
            # Randomize residue order to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                compatible = True
                for r_prev, m_prev in current_pairs:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    current_pairs.append((r, m))
                    found = True
                    break
            
            # If no residue found, we skip this modulus in this restart
        
        if len(current_pairs) > best_count:
            best_count = len(current_pairs)
            best_result = current_pairs
            # Optional: early stop if we reach a high number? 
            # Max possible is 127. Unlikely to reach.
            
    # Convert to list of lists as required
    return [[r, m] for r, m in best_result]
