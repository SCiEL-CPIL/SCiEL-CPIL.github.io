"""Rewritten program: global greedy for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import math


def construct():
    chosen = []  # list of [m, r]
    
    # Iterate moduli in increasing order to prioritize larger weights (1/m)
    for m in range(20, 201):
        # Try to find a valid residue r for this modulus m
        placed = False
        for r in range(m):
            ok = True
            for (m2, r2) in chosen:
                g = math.gcd(m, m2)
                # If residues are congruent modulo gcd, the classes intersect
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                chosen.append([m, r])
                placed = True
                break
        # If no residue works, skip this modulus
        
    return chosen
# EVOLVE-BLOCK-END
