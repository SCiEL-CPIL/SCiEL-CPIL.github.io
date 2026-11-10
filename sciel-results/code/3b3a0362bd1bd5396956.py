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
    
    # Find the smallest max_size such that all 101 residues are achievable
    # with subsets of size <= max_size
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
    # Use the smallest possible size for each residue to minimize the max row length
    
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
    
    # Verify correctness
    for r in range(101):
        row = result[r]
        assert len(row) > 0, f"Row {r} is empty"
        assert len(set(row)) == len(row), f"Row {r} has duplicates"
        assert all(1 <= x <= 10 for x in row), f"Row {r} has out-of-range values"
        computed = sum(inv[x] for x in row) % 101
        assert computed == r, f"Row {r} sums to {computed}, expected {r}"
    
    return result
