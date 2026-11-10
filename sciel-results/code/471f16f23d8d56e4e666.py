import time
from itertools import combinations

def construct():
    # Precompute modular inverses of 1..10 mod 101
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # residues_by_size[s] maps residue -> subset
    # We only need to store the first subset found for each residue/size
    residues_by_size = [dict() for _ in range(11)]
    
    # Precompute all subset sums
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            if residue not in residues_by_size[s]:
                residues_by_size[s][residue] = list(combo)
    
    # For each residue r, find the minimum size subset that achieves it.
    # This minimizes the maximum row length because each row is chosen independently
    # to be as short as possible.
    result = []
    for r in range(101):
        for s in range(1, 11):
            if r in residues_by_size[s]:
                result.append(residues_by_size[s][r])
                break
        else:
            # Fallback: should not happen as all residues are covered by size <= 5
            result.append([])
            
    return result
