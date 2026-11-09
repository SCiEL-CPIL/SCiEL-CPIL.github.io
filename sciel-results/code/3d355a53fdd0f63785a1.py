import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0

    def can_add(S, x):
        if x in S:
            return True
        S_list = sorted(S)

        # Case 1: a=x, b,c in S, x < b <= c, x | (b+c)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False

        # Case 2: b=x, a,c in S, a < x <= c, a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Case 3: c=x, a,b in S, a < b <= x, a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False

        return True

    def greedy_add(S, candidates):
        S = set(S)
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S

    best = set(range(171, 257))
    best_size = len(best)

    # Strategy 1: Greedily add from 1 to 170
    S1 = greedy_add(best, range(1, 171))
    if len(S1) > best_size:
        best = S1
        best_size = len(S1)

    # Strategy 2: Different base tails
    for start in range(155, 172):
        if time.time() > deadline - 15:
            break
        base = set(range(start, 257))
        S = greedy_add(base, range(1, start))
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Strategy 3: Greedy from high to low starting empty
    if time.time() < deadline - 30:
        S3 = greedy_add(set(), range(256, 0, -1))
        if len(S3) > best_size:
            best = S3
            best_size = len(S3)

    # Strategy 4: Greedy from low to high starting empty
    if time.time() < deadline - 30:
        S4 = greedy_add(set(), range(1, 257))
        if len(S4) > best_size:
            best = S4
            best_size = len(S4)

    # Strategy 5: Randomized greedy
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            candidates = list(range(1, 171))
            random.shuffle(candidates)
            S = greedy_add(set(range(171, 257)), candidates)
            if len(S) > best_size:
                best = S
                best_size = len(S)

    return sorted(best)
