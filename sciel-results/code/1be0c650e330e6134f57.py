import time

def construct():
    start = time.time()
    DEADLINE = start + 80.0  # safety margin under 90s

    def time_up():
        return time.time() > DEADLINE

    # We build an increasing Sidon sequence of prefix sums P_0=0 < P_1 < ... < P_n
    # Gaps a_i = P_i - P_{i-1} must be strictly increasing, each in [1,256].
    # Sidon: all pairwise differences P_j - P_i (i<j) are distinct.

    best = None
    best_len = 0

    def update_best(prefix):
        nonlocal best, best_len
        n = len(prefix) - 1  # number of gaps = number of sequence elements
        if n > best_len:
            best_len = n
            best = [prefix[i+1] - prefix[i] for i in range(n)]

    # Greedy baseline: pick smallest valid gap each step
    def greedy():
        prefix = [0]
        diffs = set()
        last_gap = 0
        while True:
            found = False
            for g in range(last_gap + 1, 257):
                pnew = prefix[-1] + g
                ok = True
                for p in prefix:
                    d = pnew - p
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    # also need pnew within reasonable bound; gaps<=256 enforced by loop
                    prefix.append(pnew)
                    for p in prefix[:-1]:
                        diffs.add(pnew - p)
                    last_gap = g
                    found = True
                    break
            if not found:
                break
        return prefix

    gprefix = greedy()
    update_best(gprefix)

    # DFS to extend beyond greedy. State: prefix list, diffs set, last_gap, length.
    # Use iterative deepening: try to reach target_len = best_len + 1, +2, ...
    # Prune: need n(n-1)/2 <= P_n (Sidon feasibility in [1,P_n])
    # Also remaining gaps must be strictly increasing and <=256.

    import sys
    sys.setrecursionlimit(100000)

    def sidon_feasible(prefix, target_len):
        # current length = len(prefix)-1; need to reach target_len gaps
        cur_len = len(prefix) - 1
        # minimal total prefix sum for target_len strictly-increasing gaps starting from last_gap
        last_gap = prefix[-1] - prefix[-2] if cur_len >= 1 else 0
        # minimal extra sum for (target_len - cur_len) more gaps, each > last_gap and increasing
        extra = 0
        g = last_gap
        for _ in range(target_len - cur_len):
            g += 1
            extra += g
        P_n_min = prefix[-1] + extra
        # Sidon: need target_len*(target_len-1)/2 <= P_n_min
        return target_len * (target_len - 1) // 2 <= P_n_min

    def dfs(prefix, diffs, last_gap, target_len):
        nonlocal best, best_len
        if time_up():
            return False
        cur_len = len(prefix) - 1
        if cur_len == target_len:
            update_best(prefix)
            return True
        # feasibility prune
        if not sidon_feasible(prefix, target_len):
            return False
        # also: max possible length from here — remaining gaps strictly increasing > last_gap, <=256
        max_extra = 0
        g = last_gap
        while g < 256:
            g += 1
            max_extra += 1
        if cur_len + max_extra < target_len:
            return False

        # candidate gaps: try a spread. Larger gaps first help Sidon distinctness.
        # We'll try from high to low but skip ones that are obviously bad.
        for g in range(256, last_gap, -1):
            if time_up():
                return False
            pnew = prefix[-1] + g
            # quick Sidon check
            ok = True
            for p in prefix:
                d = pnew - p
                if d in diffs:
                    ok = False
                    break
            if not ok:
                continue
            prefix.append(pnew)
            new_diffs = []
            for p in prefix[:-1]:
                d = pnew - p
                new_diffs.append(d)
                diffs.add(d)
            if dfs(prefix, diffs, g, target_len):
                return True
            for d in new_diffs:
                diffs.discard(d)
            prefix.pop()
        return False

    # Try to extend one at a time
    for target in range(best_len + 1, best_len + 12):
        if time_up():
            break
        if not sidon_feasible(gprefix, target):
            break
        # start DFS from greedy prefix
        prefix = gprefix[:]
        diffs = set()
        for i in range(len(prefix)):
            for j in range(i):
                diffs.add(prefix[i] - prefix[j])
        last_gap = prefix[-1] - prefix[-2]
        dfs(prefix, diffs, last_gap, target)

    if best is None:
        best = [1]
    return best
