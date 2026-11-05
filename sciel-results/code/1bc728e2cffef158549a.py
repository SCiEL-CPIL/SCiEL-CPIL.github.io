import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # All candidate moduli
    candidates = list(range(20, 201))
    
    # Greedy: process in increasing m, assign smallest valid residue
    selected = []  # list of [m, r]
    
    for m in candidates:
        if time.time() > deadline:
            break
        found = False
        for r in range(m):
            ok = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                selected.append([m, r])
                found = True
                break
        # If no residue found, skip this modulus
    
    # Local search: try to improve by reassigning residues
    # and including skipped moduli
    if time.time() < deadline - 5:
        skipped = [m for m in candidates if m not in [s[0] for s in selected]]
        for m in skipped:
            if time.time() > deadline:
                break
            # Try all residues
            for r in range(m):
                ok = True
                for m2, r2 in selected:
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    selected.append([m, r])
                    break
    
    return selected
