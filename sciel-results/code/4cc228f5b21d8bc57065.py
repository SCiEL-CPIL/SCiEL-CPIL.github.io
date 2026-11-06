import math
from itertools import combinations

def construct():
    # List of all valid moduli
    moduli = list(range(20, 201))
    # Sort by weight (1/m) descending, i.e., m ascending
    moduli.sort()
    
    selected = [] # List of (m, r)
    
    for m in moduli:
        # Try to find a residue r for m that is disjoint with all selected
        # Residues r in [0, m)
        # We can try small residues first, as they are less likely to conflict
        # with small residues of other moduli if gcd is large enough
        found = False
        for r in range(m):
            valid = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                found = True
                break
        
        # If no residue found, we skip this modulus in the greedy pass
        # A more advanced algorithm would backtrack, but let's see if greedy is enough
        
    # Convert to required format
    return [[m, r] for m, r in selected]
