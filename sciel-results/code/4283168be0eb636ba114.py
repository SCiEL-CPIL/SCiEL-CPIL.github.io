"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    # We want to maximize sum(1/m).
    # Smaller m are more valuable.
    # We process in increasing order of m.
    # This ensures we keep the high-value terms.
    
    selected = [] # List of [m, r]
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    # Greedy approach: try each modulus in order, pick the residue that allows
    # maximum future flexibility (or just the first valid one)
    # Since we want to maximize sum(1/m), we should try to include as many small moduli as possible.
    
    # A better heuristic might be to look ahead, but let's start with a simple greedy
    # that tries all residues for each modulus and picks one that is valid.
    # To maximize the score, we should prioritize including small moduli.
    
    # Let's try a simple greedy: for each m from 20 to 200, try to add it.
    # If multiple residues work, which one to pick?
    # Picking a residue that is "compatible" with many future moduli might be better.
    # For now, let's just pick the first valid residue.
    
    for m in moduli:
        if time.time() > deadline:
            break
            
        found = False
        for r in range(m):
            if is_compatible(m, r, selected):
                selected.append([m, r])
                found = True
                break
        
        # If no residue works, we skip this modulus.
        # This is a valid choice since we don't have to include all moduli.
    
    return selected
# EVOLVE-BLOCK-END
