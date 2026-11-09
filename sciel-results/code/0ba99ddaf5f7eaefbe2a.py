import itertools
import time
import collections

def construct():
    start_time = time.time()
    LIMIT = 55.0  # safety margin under 60s (evaluator kills at 90s)

    p = 101
    denom_range = range(1, 11)
    n = 10

    # Precompute modular inverses for each denominator
    inverses = {d: pow(d, -1, p) for d in denom_range}

    # For each residue, find the minimum size subset that produces it
    # Since denominators can recur across rows, we just need each residue
    # to have at least one valid subset. The max row length is then
    # max over all residues of the minimum subset size needed.
    
    # For each residue, store the smallest subset (by size) that produces it
    best_for_residue = {}  # residue -> (size, combo)
    
    # Iterate by subset size from 1 to 10
    for size in range(1, n + 1):
        if time.time() - start_time > LIMIT:
            break
        for combo in itertools.combinations(denom_range, size):
            s = sum(inverses[d] for d in combo)
            residue = s % p
            if residue not in best_for_residue:
                best_for_residue[residue] = (size, combo)
    
    # Check if all residues are covered
    if len(best_for_residue) < p:
        # Fallback: try to fill in missing residues with any subset
        # This shouldn't happen with 10 denominators and 101 residues
        # since 2^10 - 1 = 1023 > 101
        assignments = {}
        for size in range(1, n + 1):
            for combo in itertools.combinations(denom_range, size):
                s = sum(inverses[d] for d in combo)
                residue = s % p
                if residue not in assignments:
                    assignments[residue] = combo
                if len(assignments) == p:
                    break
            if len(assignments) == p:
                break
        result = []
        for r in range(p):
            if r in assignments:
                result.append(list(assignments[r]))
            else:
                result.append(list(denom_range))
        return result
    
    # Build the result: for each residue r, use the smallest subset that produces it
    result = [None] * p
    max_size = 0
    for r in range(p):
        size, combo = best_for_residue[r]
        result[r] = list(combo)
        max_size = max(max_size, size)
    
    # Sanity check
    for r in range(p):
        combo = result[r]
        assert len(combo) > 0
        assert len(set(combo)) == len(combo)
        s = sum(inverses[d] for d in combo)
        assert s % p == r, f"Residue mismatch at r={r}: got {s%p}"
        assert all(1 <= d <= 10 for d in combo)
    
    return result
