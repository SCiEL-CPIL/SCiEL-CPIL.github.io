import time, random

N = 128
DEADLINE = 50.0

def _forbidden_adds(S, x):
    lst = list(S)
    for i in range(len(lst)):
        u = lst[i]
        for j in range(i+1, len(lst)):
            v = lst[j]
            if (x+u) in S and (x+v) in S and (u+v) in S:
                return True
    return False

def _greedy(core, order, deadline):
    S = set(core)
    for x in order:
        if time.time() > deadline:
            break
        if x in S:
            continue
        if not _forbidden_adds(S, x):
            S.add(x)
    return S

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best = set(range(N//2+1, N+1))
    small = list(range(1, N//2+1))
    allv = list(range(1, N+1))
    rng = random.Random(987654321)
    it = 0
    while time.time() < deadline - 6.0:
        it += 1
        order = small[:]
        rng.shuffle(order)
        S = _greedy(best, order, deadline)
        if len(S) > len(best):
            best = S
        if it % 5 == 0:
            o2 = allv[:]
            rng.shuffle(o2)
            S2 = _greedy(set(), o2, deadline)
            if len(S2) > len(best):
                best = S2
    # local improvement
    while time.time() < deadline - 1.0:
        improved = False
        # add-all
        S = set(best)
        for x in range(1, N+1):
            if x not in S and not _forbidden_adds(S, x):
                S.add(x)
        if len(S) > len(best):
            best = S
            improved = True
        # 1-swap
        for x in list(best):
            if time.time() > deadline - 1.0:
                break
            S = set(best)
            S.discard(x)
            for y in range(1, N+1):
                if y not in S and not _forbidden_adds(S, y):
                    S.add(y)
            if len(S) > len(best):
                best = S
                improved = True
        if not improved:
            break
    return [int(v) for v in best]
