import time
from collections import defaultdict

def construct():
    P = 101
    DENOMS = list(range(1, 11))
    INV = {x: pow(x, -1, P) for x in DENOMS}
    
    # Precompute all non-empty subsets with their sum mod P and size
    # There are 2^10 - 1 = 1023 non-empty subsets
    subsets_by_residue = {r: [] for r in range(P)}
    
    for mask in range(1, 1 << 10):
        terms = [DENOMS[i] for i in range(10) if (mask >> i) & 1]
        s = sum(INV[x] for x in terms) % P
        size = len(terms)
        subsets_by_residue[s].append((size, terms))
    
    # Sort each residue's subsets by size
    for r in range(P):
        subsets_by_residue[r].sort(key=lambda x: x[0])
    
    # Find minimum k such that every residue has at least one subset of size <= k
    for k in range(1, 11):
        all_covered = True
        for r in range(P):
            if not any(sz <= k for sz, _ in subsets_by_residue[r]):
                all_covered = False
                break
        if all_covered:
            break
    
    # Assign to each residue the smallest-size subset
    result = []
    for r in range(P):
        min_size = subsets_by_residue[r][0][0]
        chosen = subsets_by_residue[r][0][1]
        result.append(chosen)
    
    return result
