import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Self-inverse elements: x^2 % MOD == 1
    self_inv = set()
    for i in range(1, N + 1):
        if i * i % MOD == 1:
            self_inv.add(i)

    # For fixed end e, the interval [s, e] is valid iff:
    # 1. No self-inverse element is in [s, e]
    # 2. For no x in [s, e] is inv[x] also in [s, e]
    #
    # Both x and inv[x] are in [s, e] iff s <= min(x, inv[x]) and max(x, inv[x]) <= e.
    # So for fixed e, we need s > min(x, inv[x]) for all pairs with max(x,inv[x]) <= e.
    # For self-inverse x <= e, we need s > x.
    #
    # bound(e) = max of all such min(x, inv[x]) and self-inverse x values.
    # s_min(e) = bound(e) + 1, length = e - bound(e).
    #
    # Compute bound(e) incrementally: as e increases, new constraints appear when
    # e reaches max(x, inv[x]) for some pair, or when e reaches a self-inverse value.

    conflict_bound = [0] * (N + 1)
    current_bound = 0

    for e in range(1, N + 1):
        if e in self_inv:
            current_bound = max(current_bound, e)
        else:
            ie = inv[e]
            if ie <= e:
                # Pair (e, ie) with ie < e: constraint is s > ie
                current_bound = max(current_bound, ie)
            # If ie > e: constraint will appear when we reach ie
        
        conflict_bound[e] = current_bound

    best_start = 2
    best_len = 1

    for e in range(1, N + 1):
        s_min = conflict_bound[e] + 1
        if s_min > e:
            continue
        length = e - s_min + 1
        if length > best_len:
            best_len = length
            best_start = s_min

    # Verify the result is valid (safety check)
    s = best_start
    e = s + best_len - 1
    valid = True
    for x in range(s, e + 1):
        for y in range(x, e + 1):
            if x * y % MOD == 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: brute force search for the longest valid interval
        best_start = 2
        best_len = 1
        for s in range(1, N + 1):
            if s in self_inv:
                continue
            for e in range(s, N + 1):
                if e in self_inv:
                    break
                # Check if interval [s, e] is valid
                # Quick check: is there any x in [s,e] with inv[x] in [s,e]?
                conflict = False
                for x in range(s, e + 1):
                    if inv[x] >= s and inv[x] <= e:
                        conflict = True
                        break
                if conflict:
                    break
                length = e - s + 1
                if length > best_len:
                    best_len = length
                    best_start = s
            if time.time() > deadline - 5:
                break

    return [best_start, best_len]
