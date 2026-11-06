import math
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    
    # Precompute gcds
    gcd_table = {}
    for i in range(2, MAX_M + 1):
        for j in range(2, MAX_M + 1):
            gcd_table[(i, j)] = _gcd(i, j)
            
    best_count = 0
    best_selection = []
    
    # Try a few different strategies
    strategies = [
        lambda m: random.shuffle(m),
        lambda m: m.sort(key=lambda x: -len([d for d in range(2, x) if x % d == 0])), # Ascending divisor count?
        lambda m: m.sort(key=lambda x: x), # Ascending value
        lambda m: m.sort(key=lambda x: -x), # Descending value
    ]
    
    for _ in range(20):
        current_moduli = moduli.copy()
        if _ % 2 == 0:
            random.shuffle(current_moduli)
        else:
            current_moduli.sort(key=lambda x: random.random())
            
        selected = []
        
        for m in current_moduli:
            valid_residues = []
            for r in range(m):
                valid = True
                for r_j, m_j in selected:
                    g = gcd_table[(m, m_j)]
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    valid_residues.append(r)
            
            if valid_residues:
                r = random.choice(valid_residues)
                selected.append([r, m])
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selection = selected
            
    return best_selection
