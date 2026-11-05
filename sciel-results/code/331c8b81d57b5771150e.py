import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    def can_add(S, x):
        if x in S:
            return False
        for a in S:
            if a >= x:
                continue
            target = (-x) % a
            for c in S:
                if c >= x and c % a == target:
                    return False
            for b in S:
                if a < b <= x and b % a == target:
                    return False
        return True

    best_S = set(range(192, 257))
    best_size = len(best_S)

    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(restart * 104729 + 7)

        tail = rng.randint(40, 95)
        S = set(range(n - tail + 1, n + 1))
        S_list = sorted(S)
        S2 = set()
        for x in S_list:
            if can_add(S2, x):
                S2.add(x)
        S = S2

        cands = [x for x in range(1, n - tail + 1) if x not in S]
        rng.shuffle(cands)
        for x in cands:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)

        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            added = False
            cands = [x for x in range(1, n + 1) if x not in S]
            rng.shuffle(cands)
            for x in cands:
                if can_add(S, x):
                    S.add(x)
                    added = True
            if not added:
                break

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    return sorted(best_S)
