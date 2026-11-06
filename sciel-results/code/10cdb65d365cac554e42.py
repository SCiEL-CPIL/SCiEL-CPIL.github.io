import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 85.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Build mapping: residue -> list of (length, sum_denoms, terms)
    residue_options = defaultdict(list)

    # Enumerate all 1023 non-empty subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        sum_denoms = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
                sum_denoms += x
        length = len(terms)
        residue_options[s].append((length, sum_denoms, terms))

    # Sort each residue's options by length, then sum_denoms
    for r in range(P):
        if r in residue_options:
            residue_options[r].sort(key=lambda t: (t[0], t[1]))

    # Determine the minimum possible max row length by binary search
    # For a given max_len L, check if every residue has at least one option with length <= L
    def feasible(max_len):
        for r in range(P):
            if r not in residue_options:
                return False
            # Find first option with length <= max_len
            found = False
            for opt in residue_options[r]:
                if opt[0] <= max_len:
                    found = True
                    break
            if not found:
                return False
        return True

    # Binary search for minimum L
    lo, hi = 1, 10
    best_L = 10
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            best_L = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Now assign: for each residue, pick the best (shortest, then smallest sum) option
    # with length <= best_L
    result = []
    for r in range(P):
        options = residue_options.get(r, [])
        if not options:
            result.append([1])
            continue
        # Find the best option with length <= best_L
        best_opt = None
        for opt in options:
            if opt[0] <= best_L:
                best_opt = opt
                break
        if best_opt is None:
            # Fallback: take the shortest available
            best_opt = options[0]
        result.append(best_opt[2])

    return result
