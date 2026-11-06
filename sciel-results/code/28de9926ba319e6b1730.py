import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Identify self-inverse elements (x * x % MOD == 1)
    self_inverse = set()
    for i in range(1, N + 1):
        if inv[i] == i:
            self_inverse.add(i)

    # Build list of 2-cycles (i, inv[i]) where i < inv[i]
    # Each such pair creates a conflict if both are in the interval
    pairs = []
    seen = set()
    for i in range(1, N + 1):
        if i in self_inverse:
            continue
        j = inv[i]
        if i < j and i not in seen:
            pairs.append((i, j))
            seen.add(i)
            seen.add(j)

    # A valid interval [s, r] must:
    # 1. Not contain any self-inverse element
    # 2. Not contain both elements of any pair (i, j) with i < j
    #    i.e., for each pair (i, j), NOT (s <= i AND j <= r)
    #    Equivalently: s > i OR r < j

    # Strategy: Two-pointer sweep
    # For each right endpoint r, find the minimum valid start s.
    # When we extend r, we may need to advance s to avoid conflicts.

    # For efficiency, precompute for each position r:
    # - If r is self-inverse, it blocks any interval containing it
    # - For each pair (i, j) with j <= r, we need s > i (if i is still in range)

    # Better approach: 
    # For a fixed r, the constraints on s are:
    #   s > max{ i : (i, j) in pairs, j <= r, i >= s }  -- circular
    #   s > max{ x : x in self_inverse, x <= r, x >= s } -- circular
    #   Also s > max{ x : x in self_inverse, x <= r } if x >= s... 
    
    # Let's use a different formulation:
    # For fixed r, define conflict_start[r] = the minimum s such that [s, r] is valid.
    # 
    # An interval [s, r] is invalid if:
    #   - There exists x in [s, r] with x self-inverse
    #   - There exists pair (i, j) with i < j, s <= i and j <= r
    #
    # For fixed r, the second condition means: there exists pair (i, j) with j <= r and i >= s.
    #   i.e., max{ i : (i,j) in pairs, j <= r } >= s
    #   So we need s > max{ i : (i,j) in pairs, j <= r }
    #
    # For the first condition: s > max{ x : x in self_inverse, x <= r }
    #   Wait, that's not right. We need NO self-inverse in [s, r].
    #   So s > max{ x : x in self_inverse, s <= x <= r }
    #   This is circular again. But we can handle it by noting:
    #   If there's a self-inverse element x in [1, r], then s must be > x for x not to be in [s,r].
    #   The strongest constraint is s > max{ x : x in self_inverse, x <= r }... 
    #   No wait. If x is self-inverse and x <= r, then to exclude x from [s,r], we need s > x.
    #   But if there are multiple self-inverse elements, we need s > ALL of them that are <= r.
    #   So s > max{ x : x in self_inverse, x <= r }.
    #   But wait, that's only if s <= x. If s > x already, x is not in [s,r].
    #   Actually: x is in [s,r] iff s <= x <= r. To avoid this, we need NOT(s <= x), i.e., s > x.
    #   So for each self-inverse x <= r, we need s > x.
    #   The binding constraint is s > max{ x : x in self_inverse, x <= r }.
    #
    # Similarly for pairs: for each pair (i,j) with j <= r, we need s > i (to avoid i being in [s,r] when j is also in [s,r]).
    #   Wait: the conflict is when BOTH i and j are in [s,r]. j <= r is given. i is in [s,r] iff s <= i.
    #   So conflict iff s <= i. To avoid: s > i.
    #   So for each pair (i,j) with j <= r, we need s > i.
    #   Binding constraint: s > max{ i : (i,j) in pairs, j <= r }.
    #
    # Therefore, for fixed r:
    #   min_valid_s = max( max_self_inv, max_pair_i ) + 1
    #   where max_self_inv = max{ x : x in self_inverse, x <= r } (or 0 if none)
    #         max_pair_i = max{ i : (i,j) in pairs, j <= r } (or 0 if none)
    #
    # And the interval [min_valid_s, r] is valid, with length r - min_valid_s + 1 (if positive).

    # Precompute:
    # For each r, we need:
    #   A[r] = max{ x : x in self_inverse, x <= r }
    #   B[r] = max{ i : (i,j) in pairs, j <= r }

    # A[r]: running max of self-inverse positions up to r
    A = [0] * (N + 1)
    current_max = 0
    for r in range(1, N + 1):
        if r in self_inverse:
            current_max = r
        A[r] = current_max

    # B[r]: for each pair (i, j), it contributes i to B[r] for all r >= j.
    # So B[r] = max{ i : (i,j) in pairs, j <= r }
    # We can compute this by sorting pairs by j and taking running max of i.
    B = [0] * (N + 1)
    # For each j, find the max i among pairs with that j
    max_i_for_j = [0] * (N + 1)
    for (i, j) in pairs:
        if i > max_i_for_j[j]:
            max_i_for_j[j] = i

    current_max = 0
    for r in range(1, N + 1):
        if max_i_for_j[r] > current_max:
            current_max = max_i_for_j[r]
        B[r] = current_max

    # Now for each r, min_valid_s = max(A[r], B[r]) + 1
    # Length = r - min_valid_s + 1 = r - max(A[r], B[r])
    # Only valid if length > 0

    best_start = 1
    best_len = 0

    for r in range(1, N + 1):
        if time.time() > deadline:
            break

        min_s = max(A[r], B[r]) + 1
        length = r - min_s + 1
        if length > best_len:
            best_len = length
            best_start = min_s

    # Fallback
    if best_len <= 0:
        for i in range(1, N + 1):
            if inv[i] != i:
                best_start = i
                best_len = 1
                break

    return [best_start, best_len]
