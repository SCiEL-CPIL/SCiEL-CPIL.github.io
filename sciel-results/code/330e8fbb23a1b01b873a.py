"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each size s from 1 to 10, compute all residues achievable
    # and store one example subset for each residue
    residues_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            if residue not in residues_by_size[s]:
                residues_by_size[s][residue] = list(combo)
    
    # Check if we're still within time budget
    if time.time() > deadline:
        # Fallback: use size 10 for all (should not happen given tiny problem)
        return [list(range(1, 11))] * 101
    
    # Find the smallest max_size such that all 101 residues are achievable
    # with subsets of size <= max_size
    min_max_size = None
    for max_s in range(1, 11):
        all_covered = True
        for r in range(101):
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
        min_max_size = 10
    
    # Now assign each residue to a subset of size <= min_max_size
    # Use the smallest possible size for each residue to minimize the max row length
    result = []
    for r in range(101):
        assigned = False
        for s in range(1, min_max_size + 1):
            if r in residues_by_size[s]:
                result.append(residues_by_size[s][r])
                assigned = True
                break
        if not assigned:
            # Should not happen if min_max_size is correct
            result.append(list(range(1, 11)))
    
    # Verify correctness (lightweight check)
    for r in range(101):
        row = result[r]
        if len(row) == 0:
            # This would be invalid; fallback
            result[r] = [1]
        computed = sum(inv[x] for x in row) % 101
        if computed != r:
            # This should not happen; find a valid subset
            for mask in range(1, 1 << 10):
                total = 0
                terms = []
                for i in range(10):
                    if mask & (1 << i):
                        denom = i + 1
                        terms.append(denom)
                        total = (total + inv[denom]) % 101
                if total == r:
                    result[r] = terms
                    break
    
    return result
# EVOLVE-BLOCK-END
