import math
from itertools import combinations

def construct():
    # We want to maximize the number of disjoint residue classes.
    # Moduli must be distinct, in [2, 128].
    # Classes [r, m] are disjoint iff r1 != r2 mod gcd(m1, m2).
    
    # Strategy:
    # 1. Consider candidate moduli. Multiples of 2 are a good start.
    # 2. Use a backtracking search to assign residues and select moduli.
    
    candidates = list(range(2, 129))
    
    best_set = []
    
    # To speed up, we can restrict to multiples of 2 first, as coprime moduli are forbidden.
    # Actually, we can mix if we are careful, but multiples of 2 is the largest pool.
    # Let's try to find a large subset of even numbers.
    
    def is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    # Backtracking search
    # We iterate through moduli, try to add them with some residue.
    # To maximize count, we should try to add as many as possible.
    
    # A simpler heuristic: 
    # Start with empty set.
    # For each modulus m in sorted order:
    #   Try all residues r for m.
    #   If r is disjoint from all existing, add it.
    #   If multiple r work, pick one (e.g., smallest).
    # This greedy might not be optimal but is fast.
    
    # Let's try a slightly better greedy: 
    # Sort moduli by something? 
    # Maybe just iterate 2..128.
    
    selected = []
    
    for m in range(2, 129):
        found = False
        for r in range(m):
            valid = True
            for r2, m2 in selected:
                if not is_disjoint(r, m, r2, m2):
                    valid = False
                    break
            if valid:
                selected.append([r, m])
                found = True
                break
        # If no residue found for m, skip m.
        
    return selected
