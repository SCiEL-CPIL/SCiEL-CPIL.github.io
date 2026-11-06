import time
from math import gcd

def construct():
    deadline = time.time() + 80.0
    MAX_M = 128
    
    # Candidate moduli 2..128
    # Two classes (r1,m1),(r2,m2) disjoint iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2)
    
    # Strategy: try to find a large clique.
    # First, try structured approaches, then search.
    
    # Approach 1: all multiples of k, need distinct residues mod k
    # For k=12: moduli 12,24,...,120 -> 10 moduli, 10 distinct residues mod 12
    best = None
    best_count = 0
    
    # Structured: multiples of k
    for k in range(2, MAX_M + 1):
        if time.time() > deadline:
            break
        moduli = [m for m in range(k, MAX_M + 1, k)]
        n = len(moduli)
        if n <= best_count:
            continue
        # Need n distinct residues mod k, so need n <= k
        if n > k:
            continue
        # Assign residues 0,1,...,n-1 to moduli
        # Check all pairs
        valid = True
        assignment = []
        for i, m in enumerate(moduli):
            r = i  # residue i mod k, but must be < m
            if r >= m:
                r = r % m
            assignment.append([r, m])
        for i in range(len(assignment)):
            for j in range(i+1, len(assignment)):
                r1, m1 = assignment[i]
                r2, m2 = assignment[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(assignment) > best_count:
            best_count = len(assignment)
            best = assignment[:]
    
    # Approach 2: greedy search with backtracking on subsets
    # Try to add moduli one by one, checking feasibility
    # Use the fact that we want distinct moduli from 2..128
    
    # Greedy: sort moduli by some heuristic, try to assign
    all_moduli = list(range(2, MAX_M + 1))
    
    # Try: pick moduli that share a large common factor structure
    # Greedy clique: start with a modulus, add others that can coexist
    
    # For a given set of moduli, can we assign residues?
    # This is a constraint satisfaction: for each pair (i,j), r_i mod gcd(m_i,m_j) != r_j mod gcd(m_i,m_j)
    
    # Try greedy with different orderings
    for ordering in ['ascending', 'descending', 'by_max_gcd']:
        if time.time() > deadline:
            break
        if ordering == 'ascending':
            mods = sorted(all_moduli)
        elif ordering == 'descending':
            mods = sorted(all_moduli, reverse=True)
        else:
            # Sort by number of divisors or something
            mods = sorted(all_moduli, key=lambda m: -sum(1 for d in range(2,m) if m%d==0))
        
        selected = []
        residues = []
        for m in mods:
            if time.time() > deadline:
                break
            # Try to find a residue r in [0, m-1] that works with all selected
            found = False
            for r in range(m):
                ok = True
                for i, (r_i, m_i) in enumerate(zip(residues, selected)):
                    g = gcd(m, m_i)
                    if r % g == r_i % g:
                        ok = False
                        break
                if ok:
                    selected.append(m)
                    residues.append(r)
                    found = True
                    break
            if len(selected) > best_count:
                best_count = len(selected)
                best = [[r, m] for r, m in zip(residues, selected)]
    
    if best is None:
        # Fallback to original
        best = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return best
