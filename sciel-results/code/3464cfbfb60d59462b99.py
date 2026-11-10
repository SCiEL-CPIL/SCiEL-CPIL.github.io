import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each size s from 1 to 10, compute all residues achievable
    # and store one example subset for each residue
    # residues_by_size[s] = dict mapping residue -> subset (first found)
    residues_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            if residue not in residues_by_size[s]:
                residues_by_size[s][residue] = list(combo)
    
    # Now assign residues to rows, trying to minimize the max row length
    # Start with size 1, then 2, etc.
    # For each residue, pick the smallest s such that it's achievable
    
    # Strategy: assign each residue to the smallest possible s
    # But we want to minimize the MAX s used
    # So we can try: for max_size from 1 to 10, check if all 101 residues can be covered
    # using only subsets of size <= max_size
    
    # For a given max_size, we need to assign each residue r to some subset of size <= max_size
    # such that all 101 residues are covered
    
    # Let's try increasing max_size and see when we can cover all residues
    
    # First, let's see what the minimum possible max_size is
    # For each max_size, count how many residues are achievable with subsets of size <= max_size
    
    # Actually, we just need each residue to be achievable, not that we use distinct subsets
    # So the question is: what's the smallest max_size such that every residue 0..100 is achievable
    # with some subset of size <= max_size?
    
    achievable = [False] * 101
    for s in range(1, 11):
        for r in residues_by_size[s]:
            achievable[r] = True
    
    # Find the smallest max_size such that all residues are achievable
    min_max_size = None
    for max_s in range(1, 11):
        all_covered = True
        for r in range(101):
            # Check if r is achievable with size <= max_s
            found = False
            for s in range(1, max_s + 1):
                if r in residues_by_size[s]:
                    found = True
                    break
            if not found:
                all_covered = False
                break
        if all_covered:
            min_max_size = max_s
            break
    
    if min_max_size is None:
        min_max_size = 10  # should not happen
    
    # Now assign each residue to a subset of size <= min_max_size
    # To minimize the max row length, we want to use the smallest possible size for each residue
    # But we also want to balance: if many residues need size min_max_size, that's fine as long
    # as the max is min_max_size
    
    result = []
    for r in range(101):
        # Find the smallest s such that r is achievable with size s
        for s in range(1, min_max_size + 1):
            if r in residues_by_size[s]:
                result.append(residues_by_size[s][r])
                break
        else:
            # Should not happen if min_max_size is correct
            result.append(list(range(1, 11)))
    
    return result
