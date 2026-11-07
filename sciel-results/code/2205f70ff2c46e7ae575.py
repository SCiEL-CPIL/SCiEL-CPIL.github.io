import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256
    n = N

    # Precompute smallest prime factor
    spf = [0] * (n + 1)
    for i in range(2, n + 1):
        if spf[i] == 0:
            for j in range(i, n + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    # Build conflict bitsets: conflict[i] has bit (j-1) set if gcd(i,j)>1
    conflict = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i != j and gcd(i, j) > 1:
                bits |= (1 << (j - 1))
        conflict[i] = bits

    best_weight = 0
    best_set = []

    # Clique cover upper bound: group candidates by smallest prime factor
    def compute_bound(candidates_bitset):
        groups = {}
        has_one = False
        c = candidates_bitset
        while c:
            lb = c & (-c)
            v = lb.bit_length()
            if v == 1:
                has_one = True
            else:
                p = spf[v]
                if p not in groups or v > groups[p]:
                    groups[p] = v
            c ^= lb
        bound = 0
        if has_one:
            bound += 1
        for max_v in groups.values():
            bound += max_v
        return bound

    def bb(candidates, cur_weight, selected):
        nonlocal best_weight, best_set

        if time.time() > deadline:
            return

        if candidates == 0:
            if cur_weight > best_weight:
                best_weight = cur_weight
                best_set = list(selected)
            return

        bound = compute_bound(candidates)
        if cur_weight + bound <= best_weight:
            return

        # Find highest weight candidate
        c = candidates
        max_v = 0
        while c:
            lb = c & (-c)
            v = lb.bit_length()
            if v > max_v:
                max_v = v
            c ^= lb

        # Branch include
        new_cand = candidates & ~conflict[max_v] & ~(1 << (max_v - 1))
        selected.append(max_v)
        bb(new_cand, cur_weight + max_v, selected)
        selected.pop()

        # Branch exclude
        new_cand = candidates & ~(1 << (max_v - 1))
        bb(new_cand, cur_weight, selected)

    all_cand = (1 << n) - 1
    bb(all_cand, 0, [])

    if not best_set:
        return [1]

    result = sorted(best_set)
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    if not valid:
        result = [1]

    return [int(x) for x in result]
