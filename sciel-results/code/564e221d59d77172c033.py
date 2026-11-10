import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute inverses
    invs = [pow(x, -1, MOD) for x in DENOMS]
    
    # Generate all non-empty subsets with their sum mod 101
    # There are 2^10 - 1 = 1023 subsets
    subset_sums = {}  # residue -> list of (length, subset_tuple)
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                terms.append(DENOMS[i])
                s += invs[i]
        s %= MOD
        length = len(terms)
        if s not in subset_sums:
            subset_sums[s] = []
        subset_sums[s].append((length, tuple(terms)))
    
    # For each residue, sort by length
    for r in subset_sums:
        subset_sums[r].sort(key=lambda x: x[0])
    
    # The problem is: for each residue r (0..100), assign a subset such that
    # sum of inverses in the subset ≡ r (mod 101), and denominators in each row
    # are distinct (guaranteed by using subsets). Denominators CAN recur across
    # different rows.
    #
    # Since rows are independent (no conflict between rows), the optimal strategy
    # is to pick the shortest subset for each residue. The maximum row length is
    # then max over all residues of (min subset length achieving that residue).
    #
    # However, let's verify: is there any coupling? The problem says "Denominators
    # may recur across different rows, but never within one row." So each row
    # independently picks a subset of distinct denominators. There's no global
    # constraint. Thus, greedy shortest-per-residue is optimal.
    
    # Check if all residues are covered
    missing = [r for r in range(MOD) if r not in subset_sums]
    if missing:
        # Should not happen, but fallback
        pass
    
    # Build the result: for each residue, pick the shortest subset
    result = []
    for r in range(MOD):
        if r in subset_sums:
            best_len, best_subset = min(subset_sums[r], key=lambda x: x[0])
            result.append(list(best_subset))
        else:
            # Fallback: should not happen
            result.append([])
    
    return result
